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
