"""Immobilien / Mietverwaltung — Stufe a: prüfe die WEG-/Hausgeldabrechnung der
Hausverwaltung (formal, gesetzlich, Daten & Summen) und ermittle den auf den
Mieter umlagefähigen Anteil.

Aufteilung der Verantwortung:
  * Das LLM EXTRAHIERT die Positionen aus dem PDF-Text und klassifiziert jede
    Position als umlagefähig (§ 2 BetrKV) ja/nein.
  * Der CODE prüft Arithmetik und Formalien (LLM rechnet ungenau).

Kein Rechtsrat — assistierende Prüfung.
"""

from __future__ import annotations

import json
import logging
import re

from local_ai.services.summarizer import _call_openai_compatible, _call_ollama

logger = logging.getLogger(__name__)

# Deterministische Keyword-Klassifikation (zuverlässiger als das LLM für die
# Standard-Kategorien).
_UML_KEYWORDS = (
    "grundsteuer", "wasser", "abwasser", "entwässer", "entwaesser", "niederschlag",
    "kanal", "heizung", "heiz", "wärme", "waerme", "warmwasser", "brennstoff", "fernwärme",
    "fernwaerme", "aufzug", "fahrstuhl", "müll", "muell", "abfall", "straßenreinig",
    "strassenreinig", "winterdienst", "hausreinig", "gebäudereinig", "gebaeudereinig",
    "ungeziefer", "garten", "grünflä", "gruenflae", "beleuchtung", "allgemeinstrom",
    "allgemein-strom", "hausstrom", "strom allgemein", "schornstein", "kaminkehr",
    "versicherung", "hauswart", "hausmeister", "wäschepflege", "waeschepflege",
    "rauchwarnmelder", "rauchmelder",
)
_NICHT_KEYWORDS = (
    "verwalt", "instandhalt", "instandsetz", "reparatur", "wartung", "rücklage", "ruecklage",
    "instandhaltungsrückl", "bankgebühr", "bankgebuehr", "kontoführ", "kontofuehr",
    "mahn", "rechtsanwalt", "gericht", "eigentümerversamml", "eigentuemerversamml",
    "leerstand", "darlehen", "zinsen", "porto", "buchhalt", "miteigentumsanteil",
    "baumpflege", "baumfäll", "baumfaell", "feuerlöscher", "feuerloescher", "flachdach",
    # Prüfungen/Checks/Steuern/Geldverkehr = i.d.R. nicht umlagefähig
    "check", "prüf", "pruef", "kontrolle", "legionell", "sanierung", "modernis",
    "kapitalsteuer", "anschaffung", "raummiete", "geldverkehr", "tankanlage",
)
# Kabel-TV / Gemeinschaftsantenne: seit 01.07.2024 i.d.R. NICHT mehr umlagefähig
# (Wegfall des Nebenkostenprivilegs / TKG-Reform) → gesondert kennzeichnen.
_KABEL_KEYWORDS = ("kabel", "antenne", "breitband", "gemeinschaftsantenne", "fernseh", "tv-", " tv ")

_KABEL_REASON = "Kabel-TV: seit 01.07.2024 i.d.R. NICHT umlagefähig (Wegfall Nebenkostenprivileg) – prüfen"


def classify_with_reason(bezeichnung: str, kategorie: str = "") -> tuple[bool, str]:
    """Return (umlagefaehig, begruendung). Conservative: unknown → not apportionable + flag."""
    s = (bezeichnung + " " + (kategorie or "")).lower()
    if any(k in s for k in _KABEL_KEYWORDS):
        return (False, _KABEL_REASON)
    if any(k in s for k in _NICHT_KEYWORDS):
        return (False, "nicht umlagefähig (Eigentümer trägt selbst)")
    if any(k in s for k in _UML_KEYWORDS):
        return (True, "umlagefähig (§ 2 BetrKV)")
    return (False, "unklar – bitte prüfen")


def classify_umlagefaehig(bezeichnung: str, kategorie: str = "") -> bool | None:
    s = (bezeichnung + " " + (kategorie or "")).lower()
    if any(k in s for k in _KABEL_KEYWORDS) or any(k in s for k in _NICHT_KEYWORDS):
        return False
    if any(k in s for k in _UML_KEYWORDS):
        return True
    return None


# § 2 BetrKV — die umlagefähigen Betriebskostenarten (Kurzform).
UMLAGEFAEHIG = [
    "Grundsteuer / laufende öffentliche Lasten",
    "Wasserversorgung",
    "Entwässerung / Abwasser",
    "Heizung (Betrieb zentrale Heizungsanlage)",
    "Warmwasser",
    "verbundene Heizungs-/Warmwasseranlagen",
    "Aufzug",
    "Straßenreinigung und Müllbeseitigung",
    "Gebäudereinigung und Ungezieferbekämpfung",
    "Gartenpflege",
    "Beleuchtung / Allgemeinstrom",
    "Schornsteinreinigung",
    "Sach- und Haftpflichtversicherung",
    "Hauswart / Hausmeister",
    "Gemeinschaftsantenne / Breitbandkabel",
    "Einrichtungen für Wäschepflege",
    "sonstige Betriebskosten (nur wenn im Mietvertrag konkret benannt)",
]

# Typisch NICHT umlagefähig (Eigentümer trägt sie selbst).
NICHT_UMLAGEFAEHIG = [
    "Verwaltungskosten / Verwaltervergütung",
    "Instandhaltung, Instandsetzung, Reparaturen",
    "Zuführung zur Instandhaltungsrücklage",
    "Kontoführungs- / Bankgebühren der Verwaltung",
    "Kosten der Eigentümerversammlung",
    "Rechtsanwalts- / Mahn- / Gerichtskosten",
    "Leerstandskosten",
    "einmalige / nicht laufende Kosten",
]


def _extract_prompt(text: str) -> str:
    # WICHTIG: KEINE Kategorienliste mitgeben — das LLM würde sie als
    # Positionen abschreiben. Die Umlagefähigkeit klassifiziert der CODE.
    return (
        "Du extrahierst die tatsächlichen Kostenpositionen aus EINER WEG-/"
        "Hausgeldabrechnung. Gib NUR ein JSON-Objekt zurück.\n\n"
        "STRIKTE REGELN\n"
        "- Nutze die Tabelle 'Verteilungsergebnis' mit der Spalte 'Ihr Anteil'. "
        "Ignoriere die 'Einnahmen-Ausgaben-Rechnung' und die Spalte 'Objekt gesamt'/'Gesamtkosten'.\n"
        "- Extrahiere ALLE Positionen — sowohl aus dem Abschnitt 'umlagefähig (Mieter)' "
        "ALS AUCH aus dem Abschnitt 'nicht umlagefähig (Mieter)'. Lass KEINE Zeile aus, "
        "auch nicht die letzten (z. B. Verwaltervergütung, Instandhaltungskosten, Anschaffungen, Steuern).\n"
        "- Als 'bezeichnung' den KONTO-/Kostennamen nehmen (z. B. 'Wartung Garagentor', "
        "'Hausmeisterkosten'), NICHT den Umlageschlüssel-Namen (z. B. 'Miteigentumsanteil', 'Anzahl Einheit').\n"
        "- ERFINDE KEINE Positionen, ergänze keine Standard-Kategorien.\n"
        "- Jede Kostenposition GENAU EINMAL. Keine Duplikate.\n"
        "- Nimm KEINE Zwischensummen/Gesamtsummen/Vorauszahlungen als Position.\n"
        "- Pro Position der Betrag aus 'Ihr Anteil' in Euro (nicht die Hausgesamtkosten).\n"
        "- Beträge als Dezimalzahl (Punkt-Dezimaltrenner, kein €, keine Tausenderpunkte).\n\n"
        "JSON-Struktur:\n"
        "{\n"
        '  "zeitraum_von": "YYYY-MM-DD" oder "",\n'
        '  "zeitraum_bis": "YYYY-MM-DD" oder "",\n'
        '  "summe_gesamtkosten": Zahl oder null,\n'
        '  "summe_eigentuemer_anteil": Zahl oder null,\n'
        '  "eigentuemer_vorauszahlung": Zahl oder null,\n'
        '  "ergebnis_eigentuemer": Zahl oder null,\n'
        '  "positionen": [\n'
        '    {"bezeichnung": "Original-Bezeichnung", "gesamt": Zahl oder null,\n'
        '     "schluessel": "Verteilerschlüssel oder """, "anteil": Zahl}\n'
        "  ]\n"
        "}\n\n"
        "ABRECHNUNGSTEXT:\n<<<\n" + text + "\n>>>\n\nNur das JSON-Objekt:"
    )


# Beide Modelle laufen mit 32k Kontext (Granite + gpt-oss). Ein Aufruf reicht für
# ~13000 Zeichen; nur sehr lange Abrechnungen werden zerlegt.
_EXTRACT_SINGLE_CHARS = 13000
_EXTRACT_CHUNK_CHARS = 11000
_EXTRACT_MAX_CHUNKS = 6
# gpt-oss ist ein Reasoning-Modell: bei zu kleinem Budget verbraucht es alles im
# Reasoning-Kanal → leerer content → JSONDecodeError. 4000 + reasoning_effort=low
# (vom Summarizer für gpt-oss automatisch) lassen genug Platz für das JSON.
_EXTRACT_MAX_TOKENS = 4000


def _focus_relevant(text: str) -> str:
    """Behalte das 'Verteilungsergebnis' (Eigentümeranteile) und verwerfe die
    redundante 'Einnahmen-/Ausgaben-Rechnung' (Gesamtkosten) — das verkürzt den
    Input, macht die Extraktion vollständiger und vermeidet das Vermischen von
    Anteil und Gesamtkosten."""
    low = text.lower()
    for marker in ("einnahmen-ausgaben", "einnahmen- und ausgaben",
                   "einnahmen-/ausgaben", "einnahmen und ausgaben", "einnahmen-/-ausgaben"):
        idx = low.find(marker)
        if idx > 800:   # nur wenn davor genug Inhalt (das Verteilungsergebnis) steht
            return text[:idx]
    return text


def _loads_json_lenient(raw: str) -> dict:
    """Parse JSON robustly: strip <think> blocks and ``` fences, and fall back
    to the first '{' … last '}' substring if the model added stray prose."""
    s = (raw or "").strip()
    if not s:
        raise json.JSONDecodeError("Empty LLM response", s or "", 0)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    # strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # last resort: take the outermost {...}
        i, j = s.find("{"), s.rfind("}")
        if i != -1 and j > i:
            return json.loads(s[i:j + 1])
        raise


async def _extract_one(chunk: str, llm: dict) -> dict:
    prompt = _extract_prompt(chunk)
    if llm["backend"] == "openai":
        raw = await _call_openai_compatible(
            prompt, llm["openai_base_url"], llm["openai_api_key"], llm["openai_model"],
            response_format="json_object", temperature=0.0,
            max_tokens_override=_EXTRACT_MAX_TOKENS, reasoning_effort="low",
        )
    else:
        raw = await _call_ollama(prompt, llm["ollama_base_url"], llm["ollama_model"], 0.0)
    return _loads_json_lenient(raw)


def _norm_name(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


async def extract_statement(text: str, llm: dict) -> dict:
    """LLM-Extraktion der Abrechnung. Ein Aufruf wenn der Text passt, sonst
    in Stücke zerlegt. Klassifikation der Umlagefähigkeit erfolgt im Code."""
    text = _focus_relevant(text or "")
    if len(text) <= _EXTRACT_SINGLE_CHARS:
        chunks = [text]
    else:
        chunks = [text[i:i + _EXTRACT_CHUNK_CHARS]
                  for i in range(0, min(len(text), _EXTRACT_CHUNK_CHARS * _EXTRACT_MAX_CHUNKS),
                                 _EXTRACT_CHUNK_CHARS)]

    out = {
        "zeitraum_von": "", "zeitraum_bis": "",
        "summe_gesamtkosten": None, "summe_eigentuemer_anteil": None,
        "eigentuemer_vorauszahlung": None, "ergebnis_eigentuemer": None,
        "positionen": [],
    }
    seen_amount: set[tuple] = set()
    seen_name: set[str] = set()
    last_exc: Exception | None = None
    ok = 0
    for ch in chunks:
        try:
            data = await _extract_one(ch, llm)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Immo extract chunk failed: %s", exc)
            continue
        ok += 1
        for k in ("zeitraum_von", "zeitraum_bis"):
            if not out[k] and data.get(k):
                out[k] = data[k]
        for k in ("summe_gesamtkosten", "summe_eigentuemer_anteil",
                  "eigentuemer_vorauszahlung", "ergebnis_eigentuemer"):
            if out[k] is None and _num(data.get(k)) is not None:
                out[k] = _num(data.get(k))
        for p in data.get("positionen", []) or []:
            bez = (p.get("bezeichnung") or "").strip()[:120]
            anteil = _num(p.get("anteil")) or 0.0
            # 0-€-Positionen sind Rauschen (oft echo der Kategorienliste) → weg.
            if not bez or anteil == 0.0:
                continue
            nm = _norm_name(bez)
            akey = (nm, round(anteil, 2))
            if akey in seen_amount or nm in seen_name:
                continue
            seen_amount.add(akey)
            seen_name.add(nm)
            # Umlagefähigkeit deterministisch klassifizieren (Code, nicht LLM).
            kw, reason = classify_with_reason(bez)
            out["positionen"].append({
                "bezeichnung": bez,
                "gesamt": _num(p.get("gesamt")),
                "schluessel": (p.get("schluessel") or "").strip()[:60],
                "anteil": anteil,
                "kategorie": "",
                "umlagefaehig": bool(kw),
                "begruendung": reason,
            })

    if ok == 0 and last_exc is not None:
        raise last_exc
    return out


def _num(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        if isinstance(v, str):
            v = v.replace(" ", "").replace("€", "").replace(" ", "")
            # tolerate German "1.234,56"
            if "," in v and "." in v:
                v = v.replace(".", "").replace(",", ".")
            elif "," in v:
                v = v.replace(",", ".")
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def _months_between(von: str, bis: str) -> int | None:
    try:
        from datetime import date
        a = date.fromisoformat(von)
        b = date.fromisoformat(bis)
        return (b.year - a.year) * 12 + (b.month - a.month) + 1
    except Exception:  # noqa: BLE001
        return None


def run_checks(extracted: dict, unit: dict | None = None) -> dict:
    """Code-basierte Arithmetik- und Formalprüfung. Kein LLM.

    Returns {findings, apportionable_total, apportionable, non_apportionable, summary}.
    """
    findings: list[dict] = []
    positions = extracted.get("positionen", [])

    apportionable = [p for p in positions if p["umlagefaehig"]]
    non_apportionable = [p for p in positions if not p["umlagefaehig"]]
    appt_total = round(sum(p["anteil"] for p in apportionable), 2)
    all_total = round(sum(p["anteil"] for p in positions), 2)

    # 1. Arithmetik: Summe der Anteile vs. ausgewiesene Eigentümer-Gesamtsumme
    stated = extracted.get("summe_eigentuemer_anteil")
    if stated is not None and positions:
        diff = round(all_total - stated, 2)
        if abs(diff) <= 1.0:
            findings.append(_f("ok", "Summen stimmen",
                               f"Summe der Positionen ({all_total:.2f} €) entspricht der ausgewiesenen Gesamtsumme ({stated:.2f} €)."))
        elif diff < 0:
            # Weniger als ausgewiesen → es fehlen vermutlich Positionen.
            findings.append(_f("warn", "Möglicherweise fehlen Positionen",
                               f"Summe der erfassten Positionen {all_total:.2f} € liegt {abs(diff):.2f} € UNTER der "
                               f"ausgewiesenen Gesamtsumme {stated:.2f} €. Vermutlich wurden nicht alle Zeilen erkannt — "
                               f"bitte mit der Abrechnung abgleichen und fehlende Positionen ergänzen (Button + Zeile)."))
        else:
            findings.append(_f("error", "Summen weichen ab",
                               f"Summe der erfassten Positionen {all_total:.2f} € liegt {diff:.2f} € ÜBER der "
                               f"ausgewiesenen Gesamtsumme {stated:.2f} € — evtl. Doppelerfassung oder falsche Beträge. Bitte prüfen."))

    # 2. Abrechnungszeitraum ≤ 12 Monate
    von, bis = extracted.get("zeitraum_von"), extracted.get("zeitraum_bis")
    if von and bis:
        m = _months_between(von, bis)
        if m is None:
            pass
        elif m > 12:
            findings.append(_f("error", "Abrechnungszeitraum > 12 Monate",
                               f"Der Zeitraum umfasst {m} Monate. Zulässig sind maximal 12 Monate (§ 556 Abs. 3 BGB)."))
        else:
            findings.append(_f("ok", "Abrechnungszeitraum zulässig", f"{von} bis {bis} ({m} Monate)."))
    else:
        findings.append(_f("warn", "Abrechnungszeitraum unklar",
                           "Kein eindeutiger Abrechnungszeitraum erkannt — bitte ergänzen (max. 12 Monate)."))

    # 3. Verteilerschlüssel je Position vorhanden?
    ohne_schluessel = [p["bezeichnung"] for p in positions if not p["schluessel"]]
    if ohne_schluessel:
        findings.append(_f("warn", "Verteilerschlüssel fehlt",
                           "Ohne erläuterten Verteilerschlüssel: " + ", ".join(ohne_schluessel[:6]) +
                           (" …" if len(ohne_schluessel) > 6 else "") +
                           ". Der Schlüssel muss nachvollziehbar angegeben sein."))

    # 4. Nicht umlagefähige Posten gefunden → dürfen NICHT an den Mieter
    if non_apportionable:
        findings.append(_f("warn", f"{len(non_apportionable)} nicht umlagefähige Position(en)",
                           "Diese gehören NICHT in die Mieterabrechnung: " +
                           ", ".join(p["bezeichnung"] for p in non_apportionable[:8]) + "."))

    # 5. Heizung/Warmwasser → separate Heizkostenabrechnung nach HeizkostenV
    heiz = [p for p in apportionable if any(k in (p["kategorie"] + p["bezeichnung"]).lower()
                                            for k in ("heiz", "warmwasser", "wärme"))]
    if heiz:
        findings.append(_f("info", "Heiz-/Warmwasserkosten enthalten",
                           "Heizung/Warmwasser müssen zu 50–70 % verbrauchsabhängig abgerechnet werden "
                           "(HeizkostenV) — i. d. R. über die separate Heizkostenabrechnung (Techem/ista). "
                           "Verbrauchswerte des Mieters bereithalten."))

    # 6. Kabel-TV / Gemeinschaftsantenne (Gesetzesänderung 2024)
    kabel = [p for p in positions
             if any(k in p["bezeichnung"].lower() for k in _KABEL_KEYWORDS)]
    if kabel:
        findings.append(_f("warn", "Kabel-TV / Antenne enthalten",
                           "Seit 01.07.2024 sind Kabel-TV-Kosten über den Sammelvertrag i. d. R. "
                           "NICHT mehr auf den Mieter umlagefähig (Wegfall des Nebenkostenprivilegs). "
                           "Betrifft: " + ", ".join(p["bezeichnung"] for p in kabel) +
                           ". Nur eine echte WEG-eigene Gemeinschaftsantenne kann noch umlagefähig sein — bitte prüfen."))

    summary = (
        f"{len(positions)} Positionen erkannt · {len(apportionable)} umlagefähig "
        f"({appt_total:.2f} €) · {len(non_apportionable)} nicht umlagefähig."
    )

    return {
        "findings": findings,
        "apportionable": apportionable,
        "non_apportionable": non_apportionable,
        "apportionable_total": appt_total,
        "summary": summary,
        "meta": {
            "zeitraum_von": von, "zeitraum_bis": bis,
            "summe_eigentuemer_anteil": stated,
            "eigentuemer_vorauszahlung": extracted.get("eigentuemer_vorauszahlung"),
            "ergebnis_eigentuemer": extracted.get("ergebnis_eigentuemer"),
        },
    }


def _f(severity: str, title: str, detail: str) -> dict:
    return {"severity": severity, "title": title, "detail": detail}
