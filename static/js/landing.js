// ============================================================================
// LANDING PAGE JAVASCRIPT
// ============================================================================

let heroChart, trainingChart;

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Landing page loaded');

    lottie.loadAnimation({
        container: document.getElementById("dp-animation"),
        renderer: "svg",
        loop: true,
        autoplay: true,
        path: "/static/images/Airplane.json"
    });


    initTrainingChart();
    animateMetrics();
});


// ============================================================================
// TRAINING CHART - Performance Over Episodes
// ============================================================================
function initTrainingChart() {
    const ctx = document.getElementById('training-chart');
    if (!ctx) return;
    
    // Real training data from evaluation_log.txt (after line 146)
    const episodes = [50, 100, 150, 200, 250];
    const avgRewards = [1586.12, 1653.98, 1505.29, 1619.91, 1607.39];
    const avgRevenues = [2212221, 2232882, 2086292, 2223806, 2220187];
    const loadFactors = [99.2, 99.4, 95.8, 99.0, 97.8];
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: episodes.map(ep => `Episode ${ep}`),
            datasets: [
                {
                    label: 'Avg Revenue (₹ Million)',
                    data: avgRevenues.map(r => r / 1000000),
                    borderColor: '#4ade80',
                    backgroundColor: 'rgba(74, 222, 128, 0.1)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 3,
                    yAxisID: 'y',
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: 'Load Factor (%)',
                    data: loadFactors,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 3,
                    yAxisID: 'y1',
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: 'Avg Reward',
                    data: avgRewards,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 3,
                    yAxisID: 'y2',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#cbd5e1',
                        font: { size: 14, weight: '600' },
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f8fafc',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(102, 126, 234, 0.5)',
                    borderWidth: 2,
                    padding: 15,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.dataset.yAxisID === 'y') {
                                label += '₹' + context.parsed.y.toFixed(2) + 'M';
                            } else if (context.dataset.yAxisID === 'y1') {
                                label += context.parsed.y.toFixed(1) + '%';
                            } else {
                                label += context.parsed.y.toFixed(2);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    position: 'left',
                    beginAtZero: false,
                    ticks: {
                        color: '#4ade80',
                        font: { weight: '600' },
                        callback: function(value) {
                            return '₹' + value.toFixed(2) + 'M';
                        }
                    },
                    grid: { 
                        color: 'rgba(74, 222, 128, 0.1)',
                        drawBorder: false
                    },
                    title: {
                        display: true,
                        text: 'Revenue (₹ Million)',
                        color: '#4ade80',
                        font: { size: 13, weight: '700' }
                    }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    min: 90,
                    max: 100,
                    ticks: {
                        color: '#f59e0b',
                        font: { weight: '600' },
                        callback: function(value) {
                            return value.toFixed(0) + '%';
                        }
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    title: {
                        display: true,
                        text: 'Load Factor (%)',
                        color: '#f59e0b',
                        font: { size: 13, weight: '700' }
                    }
                },
                y2: {
                    display: false,
                    min: 1400,
                    max: 1700
                },
                x: {
                    ticks: { 
                        color: '#cbd5e1',
                        font: { weight: '600' }
                    },
                    grid: { 
                        color: 'rgba(203, 213, 225, 0.05)',
                        drawBorder: false
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeInOutQuart'
            }
        }
    });
}

// ============================================================================
// ANIMATE METRICS ON SCROLL
// ============================================================================
function animateMetrics() {
    const metrics = document.querySelectorAll('.metric-value');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const text = target.textContent;
                
                // Extract number from text
                const match = text.match(/[\d,.]+/);
                if (match) {
                    const value = parseFloat(match[0].replace(/,/g, ''));
                    const prefix = text.split(match[0])[0];
                    const suffix = text.split(match[0])[1] || '';
                    
                    animateValue(target, 0, value, 1500, prefix, suffix);
                }
                
                observer.unobserve(target);
            }
        });
    }, { threshold: 0.5 });
    
    metrics.forEach(metric => observer.observe(metric));
}

function animateValue(element, start, end, duration, prefix = '', suffix = '') {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        
        // Format based on value
        let displayValue;
        if (end >= 1000000) {
            displayValue = (current / 1000000).toFixed(2) + 'M';
        } else if (end >= 1000) {
            displayValue = current.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
        } else {
            displayValue = current.toFixed(1);
        }
        
        element.textContent = prefix + displayValue + suffix;
    }, 16);
}

// ============================================================================
// NAVIGATION
// ============================================================================
function goToSimulation() {
    // Navigate to control page
    window.location.href = '/control';
}

// ============================================================================
//Performance Log Viewer
// ============================================================================
function openLogViewer() {
    document.getElementById('logOverlay').style.display = 'block';
    document.getElementById('logPanel').classList.add('active');
    loadEvaluationLogs();
}

function closeLogViewer() {
    document.getElementById('logOverlay').style.display = 'none';
    document.getElementById('logPanel').classList.remove('active');
}
function loadEvaluationLogs() {
    fetch('/api/evaluation_log')
        .then(res => res.text())
        .then(text => {
            document.getElementById('logContent').textContent = text;
        })
        .catch(() => {
            document.getElementById('logContent').textContent = '❌ Failed to load logs';
        });
}



// ============================================================================
// SMOOTH SCROLLING
// ============================================================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ============================================================================
// PARALLAX EFFECT
// ============================================================================
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const bg = document.querySelector('.hero-section');
    bg.style.backgroundPositionY = `${scrolled * 0.2}px`;
});


console.log('✅ Landing page scripts initialized');