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
});
