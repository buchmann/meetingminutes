"""Multi-source consolidator: take N meeting transcripts + uploaded documents
and produce ONE structured Markdown deliverable — an executive summary, a
product specification, or a project specification.

Pipeline:
    1. Caller assembles a list of "source" items from:
         - completed local-ai jobs (transcript + optional summary)
         - uploaded documents (via :mod:`document_checker`)
    2. ``build_source_pack()`` formats every item into a labeled chunk so the
       LLM knows which source each statement came from.
    3. ``consolidate()`` plugs the source pack into the right prompt template
       (chosen by output type + language) and calls the configured LLM.
       Long source packs are truncated to fit the model's context window
       using the same tokenizer-aware logic the summarizer uses.
    4. The returned Markdown is handed back to the router, which uses the
       existing ``document_checker.generate_*`` helpers to deliver it as
       .docx / .pdf / .md / .txt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from local_ai.services.summarizer import (
    _call_openai_compatible,
    _call_ollama,
    _get_model_profile,
    _truncate_transcript_to_fit,
)

logger = logging.getLogger(__name__)


# ── Output types ───────────────────────────────────────────────────────────

OUTPUT_TYPES = ("summary", "product_spec", "project_spec")

# Human-readable labels (populated from the design workflow output below)
LABELS = {
    "en": {
        "summary": "Consolidated Summary",
        "product_spec": "Product Specification",
        "project_spec": "Project Specification",
    },
    "de": {
        "summary": "Konsolidierte Zusammenfassung",
        "product_spec": "Produkt-Spezifikation",
        "project_spec": "Projekt-Spezifikation",
    },
}


# ── Prompt templates ───────────────────────────────────────────────────────
# These are the workflow-designed prompts. Each contains exactly ONE
# ``{sources}`` placeholder and produces Markdown that follows the IBM-Concert
# reference example.
# The contents below are placeholders and will be replaced by the polished
# prompts produced by the consolidator-prompts workflow.

PROMPTS: dict[str, dict[str, str]] = {
    "en": {
        "summary": """You are a senior technical writer producing a CONSOLIDATED EXECUTIVE SUMMARY in English. Your input is a bundle of meeting transcripts and uploaded documents from a single engagement or project. Your job is to fuse them into ONE coherent, sober, factual Markdown document that an executive can read in a few minutes to understand what is going on across all the meetings and materials.

OUTPUT RULES
- Output ONLY the final Markdown document. No preamble, no commentary, no code fences, no "Here is the summary".
- Write in English throughout, in third person, in an engineering / management-report register. No greetings, no sign-offs, no first-person plural cheerleading, no marketing language, no emojis.
- NEVER invent facts, names, dates, numbers, decisions, owners, or deadlines. Synthesize ONLY what is present in the sources below. If something is unclear, contradictory, or missing, move it into "Open questions" rather than guessing.
- Reference specific people, teams, tools, products, platforms, and parties by the exact names they are given in the sources (for example Ansible Automation Platform, BMC TrueSight, OPA, Git, customer names, IBM roles). Do not anonymize and do not paraphrase product names.
- Use hyphens and em-dashes for inline clarifications. Keep bullets terse - usually one line, occasionally with a short qualifier clause.
- Distinguish "decided", "proposed", and "open" precisely. Never promote a proposal to a decision.

REQUIRED STRUCTURE
Open with a title block in exactly this order:
1. H1 title of the form "<Project or Topic> - Consolidated Summary DRAFT 1" (derive the project/topic name from the sources; if no clear name exists, use the most frequently referenced subject).
2. A 1-2 line subtitle/positioning sentence describing what the initiative is and where it sits (which platforms, which customer, which program).
3. A line "Status: Consolidated summary for review".
4. A "Prepared from..." attribution line listing the kinds of source material actually present (for example "meeting minutes of <dates>, uploaded slide deck, architecture note"). Use only source descriptors visible in the input.
5. A "Date:" line with month and year if discernible from the sources; otherwise omit the date line entirely rather than invent one.

Then a single short paragraph titled exactly "How to read this document." (2-4 sentences). It must state that the document consolidates multiple meetings and documents into one narrative, that it separates Decisions from Open questions, and that the goal is to give leadership a shared factual baseline.

Then a "Contents" line followed by the numbered sections.

NUMBERED SECTIONS (tailored for a SUMMARY, not a spec)
1. Purpose and scope - 1-2 short paragraphs framing what the engagement is, who the parties are, and what the summary covers. Explicitly state what the initiative does and what it does NOT cover, if the sources say so.
2. Key participants and timeline - bullets listing the people, roles, organizations, and meeting dates that appear in the sources. Do not invent attendees.
3. Topics discussed - one subsection (3.1, 3.2, 3.3 ...) per major theme that recurs across the sources. Each subsection follows the same internal pattern:
   (a) a short framing paragraph naming the topic in plain language and referencing the customer concern or central point,
   (b) a "Decisions and confirmed points" subheading with terse declarative bullets (each starting with a noun phrase or imperative, optionally with a bolded lead-in followed by a period and a brief clarification),
   (c) an "Open questions" subheading with bullets phrased as direct questions ending with "?".
   Do NOT mix decisions and open questions in the same list.
4. Action items and next steps - bullets of the form "Owner - action - due date if stated". If owner or date is missing in the sources, write "owner TBD" or "no date" rather than guess.
5. Cross-cutting observations - a final section explicitly labeled as touching multiple prior topics (for example risks, dependencies, classification, security, staffing). Use the same Decisions / Open questions split. If no genuine cross-cutting theme is present in the sources, omit this section rather than fabricate one.

CONTENT DISCIPLINE
- Keep statements at a capability and decision level - clear enough for an executive to act on, not a verbatim transcript replay.
- When two sources disagree, surface the disagreement explicitly in Open questions, naming both positions.
- Quote a short phrase from the sources only when the exact wording matters (a decision, a deadline, a constraint). Otherwise paraphrase tightly.
- Do not include a section if the sources contain nothing for it; never pad with filler.

SOURCES
The following block contains all transcripts and documents you must consolidate. Treat it as the sole source of truth.

{sources}

Now produce the Markdown document.""",
        "product_spec": """You are a senior product manager and technical writer. Your task is to consolidate the provided source material - meeting transcripts, brainstorming notes, PoC review minutes, and uploaded documents - into a single, coherent Product Specification document in English Markdown. The output must read like an engineering-grade requirements spec, not a marketing brief.

SOURCE MATERIAL:
{sources}

OUTPUT RULES

Return ONLY the final Markdown document. No preamble, no explanation, no code fences, no commentary before or after. Do not greet the reader, do not sign off, do not include an executive summary.

CONTENT RULES

Synthesize strictly from the source material. Never invent facts, names, tools, dates, numbers, or commitments. If something is not stated or cannot be confidently inferred from the sources, do not assert it - move it into Open Questions instead. When the sources reference specific products, platforms, parties, customers, or internal teams by name (for example Ansible Automation Platform, BMC TrueSight, OPA, Git, IBM engineering, a named customer), use those exact names. Do not generalize them away.

Keep requirements at a capability level - clear enough to assess feasibility, open enough not to over-fit a single implementation or use case.

REQUIRED STRUCTURE

Open with a title block, in this order, each on its own line or short block:
- H1 document title of the form "<Product or Topic Name> Requirements Spec DRAFT 1" (use the product or topic name as it appears in the sources).
- A 1-2 line subtitle/positioning sentence stating what the system is and which platforms or context it sits on top of.
- A "Status:" line (for example "Status: Initial draft for review").
- A "Prepared from..." attribution line citing the actual source material referenced (for example PoC review minutes, brainstorming sessions, uploaded documents).
- A "Date:" line with month and year if present in the sources, otherwise omit the specific date.

Then a single short paragraph beginning "How to read this document." (2-4 sentences). It must explicitly state the Confirmed-vs-Open-Questions split, note that requirements are kept at a capability level, and state the purpose (drive the roadmap and feasibility discussion).

Then a "Contents" heading listing the numbered sections.

Then numbered top-level sections:

1. Purpose and scope - 1-2 short paragraphs framing what the product is, who it is for, and explicitly what it is NOT. If the sources position it as an umbrella, cross-cutting layer, or orchestrator over other systems, say so.

2. Foundational concepts - a small number of shared prerequisites as bullets, each with a bold or italic lead-in phrase followed by a short clarifying sentence.

3. Requirement areas - split into subsections 3.1, 3.2, 3.3, and so on, one per functional area present in the sources (for example policy and governance, resource handling, integrations, user experience, observability, data handling). Each subsection must contain, in this order:
   (a) a short framing paragraph naming the concern in plain language, referencing the customer pain or central topic where the sources support it;
   (b) a "Confirmed needs" subheading with terse declarative capability bullets, each starting with a bold or noun-phrase lead-in followed by a period and a brief clarification;
   (c) an "Open Questions / To Decide" subheading with bullets phrased as direct questions ending with "?". Each question probes a real decision (authorship, granularity, evaluation timing, scope, audit, ownership).

Within a subsection, you may also include 1-3 user stories in the form "As a <role>, I want <capability>, so that <outcome>." only when the sources clearly support the role and intent. Do not fabricate roles.

Never mix Confirmed statements and Open Questions in the same list.

4. Cross-cutting requirements - one or more sections, explicitly labeled as cutting across the prior sections. Use the same Confirmed needs / Open Questions / To Decide structure. Acceptable alternative subheading labels here: "CONFIRMED NEED" and "OPEN - TO DECIDE".

5. Non-functional requirements - capability-level bullets covering items such as performance, scalability, availability, security, compliance, auditability, where supported by the sources. Use the same Confirmed vs Open split.

6. Out of scope - a short bullet list of things the sources explicitly say the product will NOT do, or that are deferred. Do not invent exclusions.

STYLE RULES

Sober, factual, third person. Engineering-spec register. Distinguish "must", "should", and "open question" precisely. Acknowledge uncertainty explicitly rather than papering over it. Use hyphens and em-dashes for inline clarifications. No first-person plural cheerleading, no marketing adjectives, no emojis, no invented metrics. Bullets should be terse - usually one line, occasionally with a short qualifier clause.

If the sources are thin in a given area, keep Confirmed needs short and put the rest into Open Questions / To Decide. It is better to have fewer confirmed bullets than to over-assert.

Begin the document now.""",
        "project_spec": """You are a technical writer producing a Project Specification in Markdown by consolidating multiple meeting transcripts and uploaded reference documents. Your entire output must be the final Markdown document only - no preamble, no closing remarks, no code fences, no commentary about what you are doing.

SOURCE MATERIAL
The source material is provided between the markers below. It may contain raw meeting transcripts, prior drafts, slide notes, brainstorming bullets, emails, and similar artifacts. Treat all of it as the ground truth for facts.

BEGIN SOURCES
{sources}
END SOURCES

ABSOLUTE RULES
- Never invent facts, names, tools, dates, numbers, parties, or commitments. If something is not stated or clearly implied in the sources, do not write it. If it is half-stated, move it to Open Questions phrased as a real question.
- Always reference specific names, products, platforms, teams, customers, and tools exactly as they appear in the sources (for example product names, vendor names, system names, project names, role names). Do not paraphrase a real product name into a generic noun.
- Write in English throughout, sober and factual, third person, engineering-spec register. No marketing language, no first-person plural cheerleading, no greeting, no sign-off, no emojis, no executive summary fluff.
- Distinguish must / should / open question precisely. Acknowledge uncertainty explicitly rather than papering over it. Use hyphens and em-dashes for inline clarifications.
- Keep requirements at a capability level - clear enough to assess feasibility, open enough not to over-fit a single implementation.

REQUIRED DOCUMENT STRUCTURE
Produce exactly this structure, in this order:

1. Title block (no preceding text):
   - H1 line of the form: "<Product or Topic> Requirements Spec DRAFT 1" - derive the product or topic name from the sources.
   - One or two subtitle lines positioning the system: what it is and which existing platforms, tools, or processes it sits on top of or coordinates. Use the real names from the sources.
   - A line beginning "Status:" (for example "Status: Initial draft for review").
   - A line beginning "Prepared from" that attributes the source material accurately (for example "Prepared from PoC review minutes, workshop notes and requirements brainstorming sessions" - only mention source types that actually appear).
   - A line beginning "Date:" giving month and year if present in the sources; otherwise omit the date line rather than invent one.

2. A single short paragraph that begins literally with "How to read this document." It must explain the Confirmed-vs-Open-Questions split, state that requirements are kept at a capability level, and state the purpose (to drive the roadmap and feasibility discussion). Two to four sentences, no more.

3. A "Contents" heading followed by a short list of the numbered sections you will produce.

4. Section "1. Purpose and scope" - one or two short paragraphs framing the project: what it is, what problem it addresses, and importantly what it is NOT (scope boundaries, what it does not execute or replace). Use real names from the sources.

5. Section "2. Foundational concepts" - a small number (typically 3 to 6) of bullets. Each bullet starts with a bolded or italicised short lead-in phrase ending in a period, followed by one clarifying sentence. These are the shared prerequisites the rest of the spec depends on (identity, inventory, visibility, trust, data model, etc., as grounded in the sources).

6. Section "3. Requirement areas" with one numbered subsection per distinct requirement area found in the sources. Choose subsections that reflect the actual concerns raised - typical examples include objectives, scope in/out, deliverables, milestones and timeline, stakeholders and roles, risks, dependencies, governance, integrations. Do not force-fit areas that are not in the sources.

   Each subsection must follow this internal structure exactly:
   (a) A short framing paragraph (2 to 4 sentences) naming the concern in plain language and, where the sources support it, referencing the customer pain or central topic.
   (b) A subheading "Confirmed needs" (or "CONFIRMED NEED" for a single-item area) followed by terse declarative bullets. Each bullet is a capability statement, often a bolded lead-in phrase then a period and a brief clarification. No questions in this list.
   (c) A subheading "Open Questions / To Decide" (or "OPEN - TO DECIDE") followed by bullets each phrased as a direct question ending with "?". Each question probes a decision the customer or engineering team must make (authorship, granularity, evaluation timing, scope, audit, ownership, timeline). No declarative statements in this list.

7. A final numbered section for cross-cutting requirements, titled in the form "N. Cross-cutting requirement: <topic>" (or plural if several). State explicitly which earlier sections it touches. Use the same Confirmed / Open structure.

FINAL CHECKS BEFORE EMITTING
- Every Confirmed bullet is a statement; every Open Questions bullet ends with "?".
- No invented facts, dates, names, or numbers.
- No greeting, no sign-off, no meta commentary.
- Output is pure Markdown, starting with the H1 title line.""",
    },
    "de": {
        "summary": """Du bist ein praeziser technischer Redakteur. Deine Aufgabe ist es, aus mehreren Meeting-Transkripten und hochgeladenen Begleitdokumenten eine einzige, kohaerente, konsolidierte Zusammenfassung in deutscher Sprache zu erstellen. Die Zusammenfassung muss eine durchgaengige Erzaehlung ergeben - kein Protokoll und keine reine Stichpunktliste, sondern ein verdichtetes Lagebild ueber alle Quellen hinweg.

Quellenmaterial (Transkripte und Dokumente, jeweils mit Quellenkennung):

{sources}

Gib AUSSCHLIESSLICH das fertige Markdown-Dokument aus. Keine Einleitung, keine Erklaerungen, keine Code-Fences, keine Kommentare zur Erstellung. Beginne direkt mit der H1-Ueberschrift.

Sprache: durchgehend Deutsch (auch Abschnittsueberschriften). Eigennamen, Produktnamen, Toolnamen, Personennamen und etablierte Fachbegriffe bleiben im Original (z. B. Ansible Automation Platform, BMC TrueSight, Control-M, OPA, Git, VS-NfD).

Strikte Faktentreue: Erfinde nichts. Verwende ausschliesslich Inhalte, die in den Quellen oben tatsaechlich vorkommen. Wenn etwas unklar, widerspruechlich oder nur angedeutet ist, verschiebe es in den Abschnitt "Offene Punkte / Zu entscheiden" und formuliere es als Frage, statt zu raten. Nenne Personen, Teams, Tools, Termine und Entscheidungen mit den exakten Bezeichnungen, die in den Quellen verwendet werden.

Ton: nuechtern, sachlich, dritte Person. Engineering- und Protokoll-Register, keine Marketing-Sprache, keine Begruessung, keine Schlussformel, keine erste Person Plural, keine Emojis. Unterscheide praezise zwischen "bestaetigt", "in Pruefung" und "offen".

Struktur des Ausgabedokuments (exakt einhalten):

Titelblock (vor der ersten nummerierten Section):
- H1 in der Form: "# <Thema/Vorhaben> - Konsolidierte Zusammenfassung"
- Eine Untertitelzeile (1-2 Saetze), die einordnet, worum es geht und in welchem Kontext (Plattformen, Programm, Kunde) das Vorhaben steht.
- Zeile "Status: Konsolidierter Stand zur Abstimmung" (oder eine in den Quellen belegte Statusangabe).
- Zeile "Erstellt aus: ..." mit Aufzaehlung der konsolidierten Quellen (Anzahl Meetings, Dokumenttypen, Zeitraum), soweit aus den Quellen erkennbar.
- Zeile "Stand: <Monat Jahr>" - nutze das spaeteste in den Quellen belegte Datum.

Praeambel:
- Genau ein kurzer Absatz mit der Ueberschrift bzw. dem Leadsatz "Wie dieses Dokument zu lesen ist." (2-4 Saetze). Stelle klar, dass die Zusammenfassung auf Capability-Ebene konsolidiert, dass zwischen getroffenen Entscheidungen und offenen Punkten getrennt wird und dass sie Roadmap- und Machbarkeitsdiskussion vorantreiben soll.
- Danach eine Zeile "Inhalt".

Nummerierte Sections (genau diese, in dieser Reihenfolge; lasse einen Abschnitt weg, wenn die Quellen dazu nichts hergeben, statt zu erfinden):

1. Zweck und Geltungsbereich - 1-2 Absaetze: worum geht es, welches Problem soll geloest werden, was ist ausdruecklich NICHT Gegenstand. Positioniere das Vorhaben (z. B. als uebergreifender Orchestrator, als PoC-Auswertung, als Programm).

2. Teilnehmer und Beitragende - knappe Liste der in den Quellen genannten Personen, Rollen und Organisationen. Keine Mailadressen, keine Telefonnummern. Bei Unsicherheit weglassen.

3. Zeitachse und behandelte Themen - chronologische Kurzliste der Meetings/Dokumente mit Datum (sofern in den Quellen vorhanden) und einer Zeile, was dort behandelt wurde.

4. Themenbereiche - der Hauptteil. Eine Subsection 4.1, 4.2, 4.3 ... pro inhaltlich eigenstaendigem Themenbereich, der sich aus den Quellen ergibt. Jede Subsection folgt exakt dieser internen Struktur:
   (a) Ein kurzer Rahmenabsatz, der den Themenbereich in einfacher Sprache benennt und den Kern-Pain bzw. die Zielsetzung referenziert.
   (b) Subheading "Bestaetigte Entscheidungen und Erkenntnisse" - terse, deklarative Bullets auf Capability-Ebene, jeweils mit einer fettgedruckten Leadphrase, gefolgt von einer kurzen Praezisierung. Keine Fragen in dieser Liste.
   (c) Subheading "Offene Punkte / Zu entscheiden" - Bullets ausschliesslich als direkte Fragen formuliert, die mit einem Fragezeichen enden. Mische niemals Aussagen und Fragen in derselben Liste.

5. Action Items und Verantwortlichkeiten - tabellarisch oder als Bulletliste: Was, Wer (Name oder Rolle aus den Quellen), Bis wann (sofern belegt). Keine erfundenen Eigentuemer und keine erfundenen Fristen.

6. Querschnittliche Aspekte - genau wie eine Themen-Subsection aufgebaut (Rahmenabsatz, "BESTAETIGTER BEDARF", "OFFEN - ZU ENTSCHEIDEN"), aber explizit als querschnittlich gekennzeichnet und mit Verweis darauf, welche der vorherigen Sections beruehrt werden (z. B. Security/Klassifizierung, Compliance, Betrieb).

Formulierungsregeln fuer Bullets:
- Bestaetigte Bullets: kurze Nominalphrase oder Imperativ als Leadphrase (fett), Punkt, dann ein praezisierender Halbsatz. Eine Zeile, hoechstens mit kurzem Nebensatz.
- Offene Bullets: vollstaendige Frage, endet mit "?". Adressiert eine konkrete Entscheidung (Autorenschaft, Granularitaet, Zeitpunkt der Bewertung, Geltungsbereich, Audit, Verantwortung).
- Verwende Bindestriche und Gedankenstriche fuer Inline-Erlaeuterungen.

Halte die Formulierungen klar genug fuer eine Machbarkeitsbewertung, aber offen genug, um sich nicht auf eine einzelne Implementierung festzulegen. Gib jetzt das Dokument aus.""",
        "product_spec": """Du bist ein technischer Redakteur und Produkt-Architekt. Deine Aufgabe ist es, aus mehreren Meeting-Transkripten und hochgeladenen Dokumenten EINE konsolidierte Produkt-Spezifikation in deutscher Sprache zu erstellen.

QUELLENMATERIAL (Transkripte und Dokumente):
---
{sources}
---

AUSGABEFORMAT
Gib AUSSCHLIESSLICH das fertige Markdown-Dokument aus. Keine Einleitung, keine Erklaerung deiner Vorgehensweise, keine Code-Fences, kein Kommentar davor oder danach. Beginne direkt mit der H1-Ueberschrift.

SPRACHE
Das gesamte Dokument ist auf Deutsch zu verfassen. Eigennamen von Tools, Plattformen, Produkten, Personen und Organisationen werden im Original belassen (z. B. Kubernetes, OPA, Git, Ansible Automation Platform).

GRUNDREGEL - KEINE ERFINDUNGEN
Synthetisiere ausschliesslich Inhalte, die in den Quellen oben tatsaechlich vorkommen. Erfinde keine Features, keine Zahlen, keine Personen, keine Plattformen, keine Termine. Wenn ein Aspekt in den Quellen unklar oder widerspruechlich ist, verschiebe ihn in die Sektion "Offene Punkte / Zu entscheiden" und formuliere ihn als Frage - rate nicht. Wenn etwas in den Quellen gar nicht vorkommt, lass es weg.

NAMEN UND REFERENZEN
Wo immer in den Quellen konkrete Tool-Namen, Plattformen, Team- oder Personennamen, Kunden, Standorte, Schnittstellen oder Klassifikationen genannt werden, uebernimm diese exakt in das Dokument. Keine Platzhalter wie "das Tool" oder "die Plattform", wenn der Name bekannt ist.

DOKUMENTSTRUKTUR (genau in dieser Reihenfolge)

Titelblock (kein Abschnittsnummer):
- H1-Ueberschrift in der Form: "<Produkt-/Themenname> Produkt-Spezifikation DRAFT 1"
- 1-2 Zeilen Untertitel: was das Produkt ist und in welchem Umfeld es sitzt (z. B. "Cross-cutting orchestration on top of ..."). Uebernimm Positionierung aus den Quellen.
- Zeile "Status: Erster Entwurf zur Abstimmung"
- Zeile "Erstellt aus: ..." mit kurzer Auflistung der konkreten Quellen, soweit in der Quellensektion benannt (Meeting-Datum, Dokumentname, Workshop o. ae.).
- Zeile "Datum:" mit Monat und Jahr, soweit aus den Quellen ableitbar; sonst nur Monat/Jahr der juengsten Quelle.

Preamble:
Ein einziger kurzer Absatz mit der Ueberschrift bzw. dem Lead-In "Wie dieses Dokument zu lesen ist." (2-4 Saetze). Erklaert explizit die Trennung "Bestaetigte Anforderungen" vs. "Offene Punkte / Zu entscheiden", dass Anforderungen auf Capability-Ebene gehalten sind (klar genug fuer eine Machbarkeitsbewertung, offen genug, um sich nicht auf einen einzelnen Anwendungsfall festzulegen), und dass das Dokument die Roadmap- und Machbarkeitsdiskussion vorantreiben soll.

Zeile "Inhalt" (kein automatisches TOC, nur Ueberschrift).

1. Zweck und Geltungsbereich
1-2 kurze Absaetze: Was ist das Produkt, fuer wen ist es, was ist es ausdruecklich NICHT. Positioniere klar (Umbrella, Add-on, eigenstaendiges System o. ae.) - basierend auf den Quellen.

2. Zielnutzer und Nutzungskontext
Knappe Bullets: Wer nutzt das Produkt, in welcher Rolle, in welchem Kontext. Nur, was in den Quellen steht.

3. Grundlegende Konzepte
Kleine Liste fett/kursiv eingeleiteter Bullets mit je einem klaerenden Satz. Nur gemeinsame Voraussetzungen, auf denen die Feature-Bereiche aufbauen.

4. Feature-Bereiche
Ein Unterabschnitt pro Funktionsbereich (4.1, 4.2, 4.3 ...). Jeder Unterabschnitt hat exakt diese interne Struktur:
- Kurzer Einleitungs-Absatz, der den Bereich in Alltagssprache benennt und - wo aus den Quellen ableitbar - den Kern-Pain des Kunden/Nutzers referenziert.
- Unterueberschrift "Bestaetigte Anforderungen" mit terse, deklarativen Capability-Bullets. Jeder Bullet beginnt mit einer fett gesetzten Kurzform, gefolgt von einem Punkt und einem knappen Klaerungssatz. Keine Fragen in dieser Liste.
- Unterueberschrift "Offene Punkte / Zu entscheiden" mit Bullets, die als direkte Fragen formuliert sind und mit "?" enden. Keine Aussagen, nur Fragen.
Niemals Bestaetigtes und Fragen in derselben Liste mischen.

5. User Stories
Knappe Liste im Format "Als <Rolle> moechte ich <Faehigkeit>, damit <Nutzen>." Nur Stories, die sich aus den Quellen ableiten lassen. Falls keine vorliegen, schreibe einen Satz "In den Quellen wurden keine expliziten User Stories formuliert." und liste stattdessen "Abgeleitete Story-Kandidaten" als offene Punkte.

6. Nicht-funktionale Anforderungen
Gleiche Confirmed/Open-Struktur. Themen z. B. Performance, Skalierung, Verfuegbarkeit, Sicherheit, Datenschutz, Klassifikation (z. B. VS-NfD), Betrieb, Lokalisierung - jedoch nur, was in den Quellen vorkommt.

7. Out of Scope
Bullet-Liste dessen, was das Produkt ausdruecklich NICHT abdeckt - laut Quellen.

8. Querschnittliche Anforderung(en)
Letzter Abschnitt. Benenne explizit, dass dies mehrere vorherige Abschnitte beruehrt. Verwende die Labels "BESTAETIGTER BEDARF" und "OFFEN - ZU ENTSCHEIDEN" in Grossbuchstaben. Beispiele: Klassifikations-getriebene Verarbeitung, mandantenfaehige Trennung, Audit.

TON
Sachlich, dritte Person, Engineering-Spec-Register. Keine Marketing-Sprache, keine Begruessung, keine Schlussformel, keine erste Person Plural, keine Emojis. Unterscheide praezise zwischen "muss", "soll" und "offene Frage". Verwende Gedankenstriche fuer Einschuebe. Capability-Wording: klar genug fuer Machbarkeit, offen genug, um sich nicht auf eine Implementierung festzulegen.

Gib jetzt ausschliesslich das fertige Markdown-Dokument aus.""",
        "project_spec": """Du bist ein technischer Redakteur und erstellst eine PROJEKT-SPEZIFIKATION auf Deutsch im Stil einer IBM-Engineering-Anforderungsspezifikation. Die Eingabe besteht aus mehreren Meeting-Transkripten und hochgeladenen Dokumenten. Konsolidiere diese Quellen zu einem einzigen, kohaerenten Markdown-Dokument.

QUELLEN:
{sources}

AUSGABEFORMAT
Gib AUSSCHLIESSLICH das fertige Markdown-Dokument aus. Keine Vorrede, keine Erlaeuterung, keine Code-Fences, keine Meta-Kommentare. Beginne direkt mit der H1-Ueberschrift.

SPRACHE
Das gesamte Dokument ist in deutscher Sprache (Sie-Form, sachlich, dritte Person). Eigennamen von Tools, Plattformen und Parteien (z. B. Ansible Automation Platform, BMC TrueSight, OPA, Git, Concert) werden im Original belassen. Etablierte Fachbegriffe wie "Proof of Concept", "Capability-Ebene", "Umbrella-Orchestrator" duerfen englisch bleiben.

STRUKTUR (strikt einzuhalten)

1. Titelblock (kein Begruessungssatz, keine Executive Summary)
   - H1: "<Projekt-/Produktname> Anforderungsspezifikation DRAFT 1" (bzw. passender Versionsmarker aus den Quellen)
   - Eine 1-2-zeilige Positionierungszeile, die beschreibt was das System ist und auf welchen Plattformen es aufsetzt
   - "Status: Erster Entwurf zur Abstimmung" (oder genauerer Status aus den Quellen)
   - "Erstellt aus: ..." mit konkreter Nennung der Quellen (z. B. PoC-Review-Protokollen, Brainstorming-Sitzungen, hochgeladenen Dokumenten)
   - "Datum: <Monat Jahr>"

2. Praeambel
   Ein einziger kurzer Absatz mit der wortwoertlichen Ueberschrift "Wie dieses Dokument zu lesen ist." (2-4 Saetze). Erklaere darin explizit die Trennung in "Bestaetigte Anforderungen" vs. "Offene Punkte / Zu entscheiden", dass Anforderungen auf Capability-Ebene gehalten werden (klar genug fuer eine Machbarkeitsbewertung, offen genug, um sich nicht auf einen einzelnen Anwendungsfall festzulegen) und dass das Dokument die Roadmap- und Machbarkeitsdiskussion vorantreiben soll.

3. "Contents"-Zeile (kurzes Inhaltsverzeichnis der nummerierten Abschnitte).

4. Nummerierte Abschnitte:
   - "1. Zweck und Geltungsbereich" - 1-2 kurze Absaetze: was wird hier festgehalten, wie positioniert sich das Vorhaben (Umbrella/cross-cutting/ausfuehrend?), und vor allem: was tut es NICHT.
   - "2. Grundlegende Konzepte" - kleine Liste von Voraussetzungen/Fundamenten als Bullets mit fettem oder kursivem Lead-in, gefolgt von einem kurzen Klaerungssatz.
   - "3. Anforderungsbereiche" mit Unterabschnitten 3.1, 3.2, 3.3, ... - ein Unterabschnitt pro thematischem Anforderungsbereich, den die Quellen hergeben (z. B. Ziele, Scope in/out, Deliverables, Meilensteine, Stakeholder, Risiken, Abhaengigkeiten, Zeitplan - jeweils nur wenn in den Quellen substanziell adressiert).
   - Letzter Abschnitt: "4. Querschnittliche Anforderung: <Thema>" (ggf. mehrere) - explizit als querschnittlich gekennzeichnet, beruehrt mehrere vorherige Abschnitte.

5. Innere Struktur JEDES Unterabschnitts in Abschnitt 3 (und Abschnitt 4)
   (a) Ein kurzer rahmender Absatz, der die Sorge in Klartext benennt und - wo aus den Quellen erkennbar - den Kern-Pain des Kunden oder das zentrale Thema referenziert.
   (b) Unterueberschrift "Bestaetigte Anforderungen" mit knappen, deklarativen Capability-Bullets. Jeder Bullet beginnt mit einer Nominalphrase oder einem fett gesetzten Lead-in, gefolgt von Punkt und kurzer Klaerung. Beispiele aus dem Referenzstil: "Durchsetzung auf Engine-Ebene.", "Kein Start auf einer belegten Ressource.", "Prioritaet zwischen Workflows - hoeher schlaegt niedriger."
   (c) Unterueberschrift "Offene Punkte / Zu entscheiden" mit Bullets, die als echte Fragen formuliert sind und mit "?" enden. Sie adressieren Entscheidungen, die mit dem Kunden und mit IBM Engineering zu klaeren sind (Autorenschaft, Granularitaet, Auswertungszeitpunkt, Umfang, Audit).
   Mische niemals bestaetigte Aussagen und offene Fragen in derselben Liste.

REGELN
- Erfinde nichts. Synthetisiere ausschliesslich, was in den Quellen steht. Wenn etwas unklar, widerspruechlich oder nur angedeutet ist, verschiebe es in "Offene Punkte / Zu entscheiden" statt zu raten.
- Nenne konkrete Namen aus den Quellen: Personen, Rollen, Teams, Tools, Plattformen, Systeme, Termine, Versionen. Keine Platzhalter wie "der Kunde" wenn ein Name vorliegt.
- Sober, sachlicher Engineering-Spec-Register. Keine Marketing-Sprache, keine Begruessung, kein Sign-off, kein "wir", keine Emojis, keine Ausrufezeichen.
- Verwende Bindestriche und Gedankenstriche fuer Inline-Klarstellungen.
- Unterscheide "muss" / "sollte" / "offene Frage" praezise.
- Bullets sind terse - meist eine Zeile, gelegentlich mit kurzem qualifizierendem Nebensatz.
- Behalte Capability-Ebene: klar genug fuer Machbarkeitsbewertung, offen genug, um sich nicht auf eine Implementierung festzulegen.
- Falls die Quellen Klassifizierungs-, Sicherheits- oder Datenschutzanforderungen (z. B. VS-NfD) erwaehnen, behandle diese als querschnittliche Anforderung in Abschnitt 4.

Beginne jetzt mit der Ausgabe des Markdown-Dokuments.""",
    },
}


# ── Source representation ──────────────────────────────────────────────────

@dataclass
class Source:
    """One input source for the consolidator.

    ``kind``    — "transcript" (full transcript text) or "summary" (existing
                  job summary JSON dumped as text) or "document" (uploaded
                  file body).
    ``title``   — short label shown in the source pack header (filename, job
                  title, meeting date, ...).
    ``body``    — the actual text content.
    ``meta``    — optional list of "key: value" lines added to the header
                  (e.g. duration, speakers, date).
    """
    kind: str
    title: str
    body: str
    meta: list[str] = field(default_factory=list)

    def render(self, index: int) -> str:
        header_parts = [f"## Source {index}: {self.title} [{self.kind}]"]
        header_parts.extend(self.meta)
        header = "\n".join(header_parts)
        body = (self.body or "").strip()
        if not body:
            body = "(no content)"
        return f"{header}\n\n{body}\n"


def build_source_pack(sources: list[Source]) -> str:
    """Render the source list into one labeled text block."""
    if not sources:
        return ""
    parts = []
    for i, src in enumerate(sources, start=1):
        parts.append(src.render(i))
    return "\n---\n\n".join(parts)


# ── Consolidate ────────────────────────────────────────────────────────────

def _get_prompt_template(output_type: str, language: str) -> str:
    if output_type not in OUTPUT_TYPES:
        raise ValueError(f"Unknown output_type: {output_type}")
    lang = "de" if language == "de" else "en"
    template = PROMPTS[lang][output_type]
    if not template or template.startswith("__PROMPT_"):
        raise RuntimeError(
            f"Prompt template for {lang}/{output_type} not initialised "
            "(workflow output not yet integrated)"
        )
    if "{sources}" not in template:
        raise RuntimeError(f"Prompt template {lang}/{output_type} missing {{sources}} placeholder")
    return template


def get_label(output_type: str, language: str) -> str:
    """Human-readable label for the output type in the chosen language."""
    lang = "de" if language == "de" else "en"
    return LABELS.get(lang, LABELS["en"]).get(output_type, output_type)


async def consolidate(
    sources: list[Source],
    output_type: str,
    *,
    language: str = "en",
    style_profile: str | None = None,
    backend: str = "openai",
    openai_base_url: str = "",
    openai_api_key: str = "",
    openai_model: str = "",
    ollama_base_url: str = "",
    ollama_model: str = "",
    detail_level: str = "detailed",
    temperature: float | None = None,
) -> dict:
    """Consolidate multiple sources into ONE Markdown document.

    ``temperature`` is the "hallucination" dial; None uses the default 0.1
    (specs/summaries should stay grounded in the sources).

    Returns ``{"markdown": str, "language": str, "output_type": str,
    "sources_count": int, "truncated": bool, "error": str | None}``.

    The ``error`` field is non-null only when the LLM call itself failed; if
    the source pack had to be truncated to fit context, ``truncated`` is True
    but the result is still returned.
    """
    if not sources:
        return {
            "markdown": "",
            "language": language,
            "output_type": output_type,
            "sources_count": 0,
            "truncated": False,
            "error": "No sources provided.",
        }

    prompt_template = _get_prompt_template(output_type, language)
    source_pack = build_source_pack(sources)

    effective_model = openai_model or ollama_model
    effective_base_url = openai_base_url or ""

    # Reuse the tokenizer-aware truncator: prompt body acts as the
    # "transcript", template wraps it with {sources}.
    truncated_pack = await _truncate_transcript_to_fit(
        source_pack,
        prompt_template.replace("{sources}", "{transcript}"),
        effective_model,
        effective_base_url,
        backend,
        detail_level,
    )
    was_truncated = len(truncated_pack) < len(source_pack)

    prompt = prompt_template.replace("{sources}", truncated_pack)

    if style_profile:
        style_instruction = (
            "\n\nWRITING STYLE GUIDE (apply subtly to prose; keep the document "
            "structure intact):\n"
            f"{style_profile}\n"
        )
        prompt = prompt + style_instruction

    logger.info(
        "Consolidating: %d sources -> %s (%s), %d chars in pack (truncated=%s)",
        len(sources), output_type, language, len(truncated_pack), was_truncated,
    )

    try:
        if backend == "openai":
            raw = await _call_openai_compatible(
                prompt, openai_base_url, openai_api_key, effective_model,
                detail_level=detail_level,
                temperature=temperature,
            )
        else:
            raw = await _call_ollama(prompt, ollama_base_url, ollama_model, temperature)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Consolidation LLM call failed")
        return {
            "markdown": "",
            "language": language,
            "output_type": output_type,
            "sources_count": len(sources),
            "truncated": was_truncated,
            "error": f"LLM service error: {type(exc).__name__}: {exc}",
        }

    cleaned = _strip_fences(raw).strip()
    if not cleaned:
        return {
            "markdown": "",
            "language": language,
            "output_type": output_type,
            "sources_count": len(sources),
            "truncated": was_truncated,
            "error": "LLM returned an empty document.",
        }

    return {
        "markdown": cleaned,
        "language": language,
        "output_type": output_type,
        "sources_count": len(sources),
        "truncated": was_truncated,
        "error": None,
    }


def _strip_fences(text: str) -> str:
    """Strip ```markdown ... ``` or ``` ... ``` fences that some LLMs add."""
    t = text.strip()
    if t.startswith("```"):
        # Drop the opening fence line
        first_nl = t.find("\n")
        if first_nl >= 0:
            t = t[first_nl + 1 :]
        # Drop trailing fence
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    return t
