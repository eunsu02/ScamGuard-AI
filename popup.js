document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab.url;
    const resultDiv = document.getElementById('result');
    const scamTextBody = document.getElementById('scamTextBody');
    
    document.getElementById('analyzeBtn').innerText = "데이터 분석 중...";
    resultDiv.style.display = "none";

    fetch(`http://localhost:8000/youtube-scam?url=${encodeURIComponent(url)}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            document.getElementById('statusText').innerText = data.status;
            document.getElementById('probText').innerText = `예측 사기 확률: ${data.highest_probability}`;
            
            if (data.status.includes("위험")) {
                document.body.style.backgroundColor = "#ffcccc";
            } else if (data.status.includes("주의")) {
                document.body.style.backgroundColor = "#fff3cd"; 
            } else {
                document.body.style.backgroundColor = "#d4edda";
            }

            // 분석된 텍스트 노출
            scamTextBody.innerHTML = "";
            if (data.detected_scams && data.detected_scams.length > 0) {
                data.detected_scams.forEach(scam => {
                    const textDiv = document.createElement('div');
                    textDiv.style.marginBottom = "8px";
                    textDiv.innerText = scam.text;
                    scamTextBody.appendChild(textDiv);
                });
            } else {
                scamTextBody.innerText = "특이 문구가 감지되지 않았습니다.";
            }

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
        }).finally(() => { 
            document.getElementById('analyzeBtn').innerText = "분석 완료"; 
        });
});

// 웹 분석 페이지 이동 로직
document.getElementById('webLinkBtn').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab.url) {
        window.open(`http://localhost:8000/web-analysis?url=${encodeURIComponent(tab.url)}`, "_blank");
    } else {
        window.open("http://localhost:8000/web-analysis", "_blank");
    }
});