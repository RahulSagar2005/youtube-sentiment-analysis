document.addEventListener("DOMContentLoaded", async () => {
    const outputDiv = document.getElementById("output");
    const API_URL = "https://ytsentimentapi.duckdns.org"; // ✅ Your EC2 backend

    // Fetch comments via your backend (API key stays server-side)
    async function fetchComment(videoID) {
        try {
            let comments = [];
            let nextPageToken = '';
            const maxPages = 5;
            let page = 0;

            while (page < maxPages) {
                // ✅ Only append pageToken if it's non-empty
                let url = `${API_URL}/comments?video_id=${videoID}`;
                if (nextPageToken) {
                    url += `&pageToken=${encodeURIComponent(nextPageToken)}`;
                }

                const response = await fetch(url);

                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }

                const data = await response.json();

                if (data.error) {
                    throw new Error(data.error);
                }

                if (!data.items || data.items.length === 0) break;

                data.items.forEach((item) => {
                    const snippet = item.snippet.topLevelComment.snippet;
                    comments.push({
                        text: snippet.textDisplay,
                        timestamp: snippet.publishedAt,
                        authorId: snippet.authorChannelId?.value || 'unknown',
                        likeCount: snippet.likeCount || 0
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
            const response = await fetch(`${API_URL}/predict`, {
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

    // Fetch and display pie chart
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

    // Fetch and display word cloud
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

    // Fetch and display trend graph
    async function fetchAndDisplayTrendGraph(sentimentData) {
        try {
            const validData = sentimentData.filter(d => d.timestamp && d.timestamp !== '');

            if (validData.length === 0) {
                outputDiv.innerHTML += `<p style="color:orange;">Not enough timestamped data for trend graph.</p>`;
                return;
            }

            const response = await fetch(`${API_URL}/generate_trend_graph`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentiments: validData })
            });

            if (!response.ok) throw new Error(`Trend graph API error: ${response.status}`);

            const blob = await response.blob();
            const imgUrl = URL.createObjectURL(blob);
            outputDiv.innerHTML += `
                <div class="section">
                    <div class="section-title">Sentiment Trend Over Time</div>
                    <img src="${imgUrl}" alt="Trend Graph" style="width:100%; border-radius:6px;"/>
                </div>
            `;
        } catch (error) {
            console.error("Error fetching trend graph:", error);
            outputDiv.innerHTML += `<p style="color:red;">Error generating trend graph: ${error.message}</p>`;
        }
    }

    // Display top 25 comments with colour-coded sentiment badges
    function displayTop25Comments(predictions) {
        const sentimentMeta = (s) => {
            const val = parseInt(s);
            if (val === 1)  return { text: 'Positive', color: '#00cc66', bg: '#003311', icon: '😊' };
            if (val === -1) return { text: 'Negative', color: '#ff4444', bg: '#330000', icon: '😠' };
            return { text: 'Neutral', color: '#cccccc', bg: '#2a2a2a', icon: '😐' };
        };

        const top25 = predictions.slice(0, 25);

        const rows = top25.map((item, index) => {
            const meta = sentimentMeta(item.sentiment);
            // Strip HTML tags for safe display
            const cleanComment = item.comment.replace(/<[^>]*>/g, '').trim();
            return `
                <div class="top-comment-row">
                    <div class="top-comment-index">${index + 1}</div>
                    <div class="top-comment-text">${cleanComment}</div>
                    <div class="top-comment-badge" style="color:${meta.color}; background:${meta.bg}; border: 1px solid ${meta.color};">
                        ${meta.icon} ${meta.text}
                    </div>
                </div>
            `;
        }).join('');

        outputDiv.innerHTML += `
            <div class="section">
                <div class="section-title">Top 25 Comments — Sentiment</div>
                <div class="top-comments-container">
                    ${rows}
                </div>
            </div>
        `;
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
                let totalSentimentScore = 0;

                predictions.forEach((item) => {
                    const s = parseInt(item.sentiment);
                    sentimentCounts[String(item.sentiment)] = (sentimentCounts[String(item.sentiment)] || 0) + 1;
                    totalSentimentScore += s;

                    if (item.timestamp) {
                        sentimentData.push({ timestamp: item.timestamp, sentiment: s });
                    }
                });

                const totalComments = comments.length;
                const uniqueCommenters = new Set(comments.map((c) => c.authorId)).size;
                const totalWords = comments.reduce(
                    (sum, c) => sum + c.text.split(/\s+/).filter((w) => w.length > 0).length, 0
                );
                const avgWordLength = (totalWords / totalComments).toFixed(2);
                const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
                const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

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

                await fetchAndDisplayChart(sentimentCounts);
                await fetchAndDisplayWordCloud(comments);
                await fetchAndDisplayTrendGraph(sentimentData);
                displayTop25Comments(predictions);
            }
        } else {
            outputDiv.innerHTML = `<p>Not a valid YouTube video page.</p>`;
        }
    });
});