document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab.url;
    const resultDiv = document.getElementById('result');
    
    document.getElementById('analyzeBtn').innerText = "분석 중...";
    resultDiv.style.display = "none";

    fetch(`http://localhost:8000/youtube-scam?url=${encodeURIComponent(url)}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            document.getElementById('statusText').innerText = data.status;
            document.getElementById('probText').innerText = `최고 사기 확률: ${data.highest_probability}`;
            
            if (data.status.includes("위험")) document.body.style.backgroundColor = "#ffcccc";
            else if (data.status.includes("주의")) document.body.style.backgroundColor = "#fff3cd";
            else document.body.style.backgroundColor = "#d4edda";

            const reasonList = document.getElementById('reasonList');
            reasonList.innerHTML = "";
            data.detected_scams.forEach(scam => {
                scam.reason.forEach(r => {
                    const item = document.createElement('div');
                    item.style.fontSize = "12px";
                    item.innerText = `• ${r.keyword}: ${r.description}`;
                    reasonList.appendChild(item);
                });
            });
            resultDiv.style.display = "block";
        }).finally(() => { document.getElementById('analyzeBtn').innerText = "분석 완료"; });
});

// 웹 분석 페이지로 이동 시 현재 URL 전달
document.getElementById('webLinkBtn').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const currentUrl = tab.url;
    
    // URL이 있을 경우 쿼리 파라미터로 포함하여 새 창 열기
    if (currentUrl) {
        window.open(`http://localhost:8000/web-analysis?url=${encodeURIComponent(currentUrl)}`, "_blank");
    } else {
        window.open("http://localhost:8000/web-analysis", "_blank");
    }
});