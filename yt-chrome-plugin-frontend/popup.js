document.addEventListener("DOMContentLoaded", async () => {
    const outputDiv = document.getElementById("output");
    const API_KEY = 'AIzaSyD6zn4W7Ql24xF3yzhtI6sN1lTjFunwJm0';
    const API_URL = "https://ytsentimentapi.duckdns.org";

    // Fetch comments from YouTube Data API
    async function fetchComment(videoID) {
        try {
            let comments = [];
            let nextPageToken = '';
            const maxPages = 5; // fetch up to 5 pages = 500 comments
            let page = 0;

            while (page < maxPages) {
                const url = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoID}&maxResults=100&pageToken=${nextPageToken}&key=${API_KEY}`;
                const response = await fetch(url);
                const data = await response.json();

                if (!data.items || data.items.length === 0) break;

                data.items.forEach((item) => {
                    const snippet = item.snippet.topLevelComment.snippet;
                    comments.push({
                        text: snippet.textDisplay,
                        timestamp: snippet.publishedAt,
                        authorId: snippet.authorChannelId?.value || 'unknown'
                    });
                });

                nextPageToken = data.nextPageToken || '';
                if (!nextPageToken) break;
                page++;
            }

            return comments;
        } catch (error) {
            console.error("Error fetching comments:", error);
            outputDiv.innerHTML += `<p style="color:red;">Error fetching comments: ${error.message}</p>`;
            return [];
        }
    }

    // Send comments to Flask API for sentiment prediction
    async function getSentimentPredictions(comments) {
        try {
            const response = await fetch(`${API_URL}/predict_with_timestamps`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    comments: comments.map((c) => ({
                        text: c.text,
                        timestamp: c.timestamp
                    }))
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error("Error getting sentiment predictions:", error);
            outputDiv.innerHTML += `<p style="color:red;">Error during sentiment analysis: ${error.message}</p>`;
            return null;
        }
    }

    // Fetch and display pie chart from Flask API
    async function fetchAndDisplayChart(sentimentCounts) {
        try {
            const response = await fetch(`${API_URL}/generate_chart`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentiment_counts: sentimentCounts })
            });

            if (!response.ok) throw new Error(`Chart API error: ${response.status}`);

            const blob = await response.blob();
            const imgUrl = URL.createObjectURL(blob);
            outputDiv.innerHTML += `
                <div class="section">
                    <div class="section-title">Sentiment Distribution</div>
                    <img src="${imgUrl}" alt="Sentiment Pie Chart"/>
                </div>
            `;
        } catch (error) {
            console.error("Error fetching chart:", error);
            outputDiv.innerHTML += `<p style="color:red;">Error generating chart: ${error.message}</p>`;
        }
    }

    // Fetch and display word cloud from Flask API
    async function fetchAndDisplayWordCloud(comments) {
        try {
            const response = await fetch(`${API_URL}/generate_wordcloud`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ comments: comments.map((c) => c.text) })
            });

            if (!response.ok) throw new Error(`Word cloud API error: ${response.status}`);

            const blob = await response.blob();
            const imgUrl = URL.createObjectURL(blob);
            outputDiv.innerHTML += `
                <div class="section">
                    <div class="section-title">Word Cloud</div>
                    <img src="${imgUrl}" alt="Word Cloud"/>
                </div>
            `;
        } catch (error) {
            console.error("Error fetching word cloud:", error);
            outputDiv.innerHTML += `<p style="color:red;">Error generating word cloud: ${error.message}</p>`;
        }
    }

    // Fetch and display trend graph from Flask API
    async function fetchAndDisplayTrendGraph(sentimentData) {
        try {
            const response = await fetch(`${API_URL}/generate_trend_graph`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentiments: sentimentData })
            });

            if (!response.ok) throw new Error(`Trend graph API error: ${response.status}`);

            const blob = await response.blob();
            const imgUrl = URL.createObjectURL(blob);
            outputDiv.innerHTML += `
                <div class="section">
                    <div class="section-title">Sentiment Trend Over Time</div>
                    <img src="${imgUrl}" alt="Trend Graph"/>
                </div>
            `;
        } catch (error) {
            console.error("Error fetching trend graph:", error);
            outputDiv.innerHTML += `<p style="color:red;">Error generating trend graph: ${error.message}</p>`;
        }
    }

    // Display top comments by sentiment
    function displayTopComments(predictions) {
        const positive = predictions.filter((p) => parseInt(p.sentiment) === 1).slice(0, 3);
        const negative = predictions.filter((p) => parseInt(p.sentiment) === -1).slice(0, 3);

        const renderComments = (items, label, color) => {
            if (items.length === 0) return '';
            return `
                <div class="section">
                    <div class="section-title" style="color:${color};">${label}</div>
                    <ul class="comment-list">
                        ${items.map((item) => `
                            <li class="comment-item">
                                <span class="comment-sentiment">${label}:</span>
                                ${item.comment}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        };

        outputDiv.innerHTML += renderComments(positive, 'Positive', '#00cc66');
        outputDiv.innerHTML += renderComments(negative, 'Negative', '#ff4444');
    }

    // Main flow
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
        const url = tabs[0].url;

        const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
        const match = url.match(youtubeRegex);

        if (match && match[1]) {
            const videoID = match[1];

            outputDiv.innerHTML = `
                <div class="section-title">YouTube Video ID</div>
                <p>${videoID}</p>
                <p>Fetching comments...</p>
            `;

            const comments = await fetchComment(videoID);

            if (!comments || comments.length === 0) {
                outputDiv.innerHTML += `<p>No comments found for this video.</p>`;
                return;
            }

            outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis...</p>`;

            const predictions = await getSentimentPredictions(comments);

            if (predictions) {
                const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
                const sentimentData = [];

                const totalSentimentScore = predictions.reduce(
                    (sum, item) => sum + parseInt(item.sentiment),
                    0
                );

                predictions.forEach((item) => {
                    sentimentCounts[String(item.sentiment)]++;
                    sentimentData.push({
                        timestamp: item.timestamp,
                        sentiment: parseInt(item.sentiment)
                    });
                });

                const totalComments = comments.length;
                const uniqueCommenters = new Set(comments.map((c) => c.authorId)).size;
                const totalWords = comments.reduce(
                    (sum, c) => sum + c.text.split(/\s+/).filter((word) => word.length > 0).length,
                    0
                );
                const avgWordLength = (totalWords / totalComments).toFixed(2);
                const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
                const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

                // Display metrics
                outputDiv.innerHTML += `
                    <div class="section">
                        <div class="section-title">Comment Analysis Summary</div>
                        <div class="metrics-container">
                            <div class="metric">
                                <div class="metric-title">Total Comments</div>
                                <div class="metric-value">${totalComments}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-title">Unique Commenters</div>
                                <div class="metric-value">${uniqueCommenters}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-title">Avg Word Count</div>
                                <div class="metric-value">${avgWordLength}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-title">Sentiment Score</div>
                                <div class="metric-value">${normalizedSentimentScore}</div>
                            </div>
                        </div>
                    </div>
                `;

                // Display all visuals and comments
                await fetchAndDisplayChart(sentimentCounts);
                await fetchAndDisplayWordCloud(comments);
                await fetchAndDisplayTrendGraph(sentimentData);
                displayTopComments(predictions);
            }
        } else {
            outputDiv.innerHTML = `<p>Not a valid YouTube video page.</p>`;
        }
    });
});