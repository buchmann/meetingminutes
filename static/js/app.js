document.addEventListener('click', async (e) => {
    if (e.target.classList.contains('delete-btn')) {
        const jobId = e.target.dataset.jobId;
        if (!confirm('Delete this job and all its files?')) return;
        e.target.setAttribute('aria-busy', 'true');
        try {
            const resp = await fetch('/api/jobs/' + jobId, { method: 'DELETE' });
            if (resp.ok) {
                e.target.closest('tr')?.remove();
            }
        } catch (err) {
            console.error('Delete failed:', err);
        }
    }

    if (e.target.classList.contains('share-btn')) {
        const btn = e.target;
        const jobId = btn.dataset.jobId;
        const makeShared = btn.dataset.shared !== '1';  // currently private -> share it
        btn.setAttribute('aria-busy', 'true');
        try {
            const resp = await fetch('/api/jobs/' + jobId + '/share', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ shared: makeShared }),
            });
            if (resp.ok) {
                if (makeShared) {
                    btn.textContent = 'Unshare';
                    btn.dataset.shared = '1';
                    btn.classList.remove('secondary');
                } else {
                    btn.textContent = 'Share';
                    btn.dataset.shared = '0';
                    btn.classList.add('secondary');
                }
            }
        } catch (err) {
            console.error('Share toggle failed:', err);
        } finally {
            btn.removeAttribute('aria-busy');
        }
    }
});

// ── Hallucination / creativity slider (LLM temperature) ─────────────────
// Reusable control. Markup:
//   <div class="halluci" data-default="0.3">
//     <input type="range" class="halluci-slider" min="0" max="1" step="0.05">
//     <span class="halluci-value"></span>
//     <button type="button" class="halluci-reset">Reset</button>
//   </div>
// Read the chosen value with window.halluciValue(rootEl) → string (e.g. "0.30").
function setupHallucinationSliders(root) {
    const scope = root || document;
    scope.querySelectorAll('.halluci').forEach((ctrl) => {
        if (ctrl.dataset.wired === '1') return;       // idempotent
        ctrl.dataset.wired = '1';
        const slider = ctrl.querySelector('.halluci-slider');
        const valEl = ctrl.querySelector('.halluci-value');
        const resetBtn = ctrl.querySelector('.halluci-reset');
        const def = parseFloat(ctrl.dataset.default);
        const defStr = isNaN(def) ? '0.30' : def.toFixed(2);

        function render() {
            const v = parseFloat(slider.value);
            if (valEl) valEl.textContent = v.toFixed(2);
            // Mark the reset button muted when already at default.
            if (resetBtn) {
                const atDefault = Math.abs(v - def) < 1e-9;
                resetBtn.disabled = atDefault;
            }
        }
        if (!isNaN(def)) slider.value = def;           // initialise to default
        slider.addEventListener('input', render);
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                if (!isNaN(def)) slider.value = def;
                render();
            });
        }
        render();
    });
}

// Helper a page can call to read the current temperature value from its slider.
window.halluciValue = function (root) {
    const ctrl = (root || document).querySelector('.halluci');
    if (!ctrl) return '';
    const slider = ctrl.querySelector('.halluci-slider');
    return slider ? slider.value : '';
};

// Auto-wire any sliders present on the page.
document.addEventListener('DOMContentLoaded', () => setupHallucinationSliders());
