// Multi-Class Dashboard JavaScript - FIXED VERSION
let econPriceChart, busPriceChart, revenueChart;
let autoRunning = false;
let autoRunInterval;
let currentAIAction = null;
let sessionStats = {
    actions: 0,
    bookings: 0,
    rewards: []
};

document.addEventListener('DOMContentLoaded', function() {
    console.log('✅ Multi-class dashboard initialized');
    initCharts();
    loadAgentInfo();
    loadRoutes();  // Load available routes for dropdown
    updateDashboard();
    setInterval(updateDashboard, 2000);
    setInterval(getAIRecommendation, 5000);
});

function initCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: { 
            legend: { 
                labels: { color: '#cbd5e1' } 
            } 
        },
        scales: {
            y: { 
                ticks: { color: '#cbd5e1' }, 
                grid: { color: 'rgba(203, 213, 225, 0.1)' } 
            },
            x: { 
                ticks: { 
                    color: '#cbd5e1',
                    maxRotation: 45,
                    minRotation: 0
                }, 
                grid: { color: 'rgba(203, 213, 225, 0.1)' } 
            }
        }
    };

    // Economy Price Chart
    const econCtx = document.getElementById('econPriceChart').getContext('2d');
    econPriceChart = new Chart(econCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Our Economy Price',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // Business Price Chart
    const busCtx = document.getElementById('busPriceChart').getContext('2d');
    busPriceChart = new Chart(busCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Our Business Price',
                data: [],
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: chartOptions
    });

    // FIXED: Daily Revenue Chart (not cumulative)
    const revCtx = document.getElementById('revenueChart').getContext('2d');
    revenueChart = new Chart(revCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Economy Daily Revenue',
                data: [],
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: '#3b82f6',
                borderWidth: 2
            }, {
                label: 'Business Daily Revenue',
                data: [],
                backgroundColor: 'rgba(139, 92, 246, 0.6)',
                borderColor: '#8b5cf6',
                borderWidth: 2
            }]
        },
        options: {
            ...chartOptions,
            scales: {
                y: { 
                    stacked: true,
                    ticks: { color: '#cbd5e1' }, 
                    grid: { color: 'rgba(203, 213, 225, 0.1)' },
                    title: {
                        display: true,
                        text: 'Daily Revenue (₹)',
                        color: '#cbd5e1'
                    }
                },
                x: { 
                    stacked: true,
                    ticks: { 
                        color: '#cbd5e1',
                        maxRotation: 45,
                        minRotation: 0
                    }, 
                    grid: { color: 'rgba(203, 213, 225, 0.1)' } 
                }
            }
        }
    });
}

// NEW: Load available routes and populate dropdown
async function loadRoutes() {
    try {
        const response = await fetch('/api/routes');
        const data = await response.json();
        
        if (data.routes && data.routes.length > 0) {
            populateRouteDropdown(data.routes, data.current_route);
            console.log('🛣️ Loaded routes:', data.routes.length);
        }
    } catch (error) {
        console.error('Error loading routes:', error);
    }
}

// NEW: Populate route dropdown with scrollable options
function populateRouteDropdown(routes, currentRoute) {
    const selector = document.getElementById('route-selector');
    if (!selector) return;
    
    // Clear existing options
    selector.innerHTML = '';
    
    // Add options
    routes.forEach(route => {
        const option = document.createElement('option');
        option.value = route;
        option.textContent = route;
        if (route === currentRoute) {
            option.selected = true;
        }
        selector.appendChild(option);
    });
    
    // Add change event listener
    selector.addEventListener('change', function() {
        changeRoute(this.value);
    });
}

// NEW: Change route handler
async function changeRoute(route) {
    try {
        showToast(`🔄 Switching to route: ${route}...`, 'info');
        
        const response = await fetch('/api/change_route', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ route: route })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(`✅ ${data.message}`, 'success');
            
            // Reset session stats
            sessionStats = {
                actions: 0,
                bookings: 0,
                rewards: []
            };
            
            document.getElementById('actions-count').textContent = '0';
            document.getElementById('total-bookings').textContent = '0';
            document.getElementById('avg-reward').textContent = '0.00';
            
            // Clear charts
            [econPriceChart, busPriceChart, revenueChart].forEach(chart => {
                chart.data.labels = [];
                chart.data.datasets.forEach(ds => ds.data = []);
                chart.update();
            });
            
            // Update dashboard
            await updateDashboard();
            await getAIRecommendation();
            
            addLogEntry(`🛣️ Switched to route: ${route}`);
        } else {
            showToast(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error changing route:', error);
        showToast('❌ Error changing route', 'error');
    }
}

async function updateDashboard() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        
        // Update route name
        document.getElementById('route-name').textContent = data.route || 'Loading...';
        document.getElementById('days-departure').textContent = data.days_to_departure || 90;
        document.getElementById('disruption-status').textContent = data.disruption || 'None';
        
        // Economy metrics
        document.getElementById('econ-price').textContent = `₹${Math.round(data.econ_price).toLocaleString()}`;
        document.getElementById('econ-sold').textContent = `${data.econ_sold} / ${data.econ_total}`;
        document.getElementById('econ-load').textContent = `${data.econ_load_factor.toFixed(1)}%`;
        document.getElementById('econ-progress').style.width = `${Math.min(100, data.econ_load_factor)}%`;
        document.getElementById('econ-revenue').textContent = `₹${Math.round(data.econ_revenue).toLocaleString()}`;
        
        // Business metrics
        document.getElementById('bus-price').textContent = `₹${Math.round(data.bus_price).toLocaleString()}`;
        document.getElementById('bus-sold').textContent = `${data.bus_sold} / ${data.bus_total}`;
        document.getElementById('bus-load').textContent = `${data.bus_load_factor.toFixed(1)}%`;
        document.getElementById('bus-progress').style.width = `${Math.min(100, data.bus_load_factor)}%`;
        document.getElementById('bus-revenue').textContent = `₹${Math.round(data.bus_revenue).toLocaleString()}`;
        
        // Overall
        document.getElementById('total-sold').textContent = `${data.total_sold} / ${data.total_seats}`;
        document.getElementById('total-load').textContent = `${data.load_factor.toFixed(1)}%`;
        document.getElementById('total-progress').style.width = `${Math.min(100, data.load_factor)}%`;
        document.getElementById('total-revenue').textContent = `₹${Math.round(data.total_revenue).toLocaleString()}`;
        
        // Competitors
        updateCompetitorPrices('econ-competitors', data.econ_competitors);
        updateCompetitorPrices('bus-competitors', data.bus_competitors);
        
    } catch (error) {
        console.error('Error updating dashboard:', error);
    }
}

function updateCompetitorPrices(elementId, prices) {
    const container = document.getElementById(elementId);
    if (!container) return;
    
    container.innerHTML = '';
    
    for (const [airline, price] of Object.entries(prices)) {
        const item = document.createElement('div');
        item.className = 'competitor-item';
        item.innerHTML = `
            <span class="competitor-name">${airline}</span>
            <span class="competitor-price">₹${Math.round(price).toLocaleString()}</span>
        `;
        container.appendChild(item);
    }
}

async function takeAction(actionId) {
    try {
        const response = await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionId })
        });
        
        const data = await response.json();
        
        if (data.success) {
            sessionStats.actions++;
            sessionStats.bookings += data.total_bookings;
            sessionStats.rewards.push(data.reward);
            
            document.getElementById('actions-count').textContent = sessionStats.actions;
            document.getElementById('total-bookings').textContent = sessionStats.bookings;
            
            const avgReward = sessionStats.rewards.reduce((a, b) => a + b, 0) / sessionStats.rewards.length;
            document.getElementById('avg-reward').textContent = avgReward.toFixed(2);
            
            addLogEntry(data.message);
            showToast(data.message, 'success');
            
            await updateHistory();
            await updateDashboard();
            await getAIRecommendation();
            
            if (data.done) {
                showToast('✈️ Flight departed! Simulation complete.', 'info');
                stopAutoRun();
            }
        }
    } catch (error) {
        console.error('Error taking action:', error);
        showToast('❌ Error taking action', 'error');
    }
}

async function getAIRecommendation() {
    try {
        const response = await fetch('/api/ai_recommendation');
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('ai-rec-content').innerHTML = `
                <div class="error">❌ ${data.error}</div>
            `;
            return;
        }
        
        currentAIAction = data.action;
        
        const statusBadge = data.agent_status === 'trained' 
            ? '<span class="badge-trained">✓ TRAINED</span>' 
            : '<span class="badge-untrained">⚠️ UNTRAINED</span>';
        
        document.getElementById('ai-rec-content').innerHTML = `
            ${statusBadge}
            <div class="ai-action">${data.action_name}</div>
            <div class="ai-reason">${data.reason}</div>
            <div style="margin-top: 10px; color: #10b981;">
                Confidence: ${(data.confidence * 100).toFixed(0)}%
                ${data.q_value !== undefined ? ` | Q-value: ${data.q_value.toFixed(2)}` : ''}
            </div>
            <div class="market-context">
                <small>
                    E: ₹${Math.round(data.market_context.econ_price).toLocaleString()} (${data.market_context.econ_vs_market}), ${data.market_context.econ_load.toFixed(1)}% full<br>
                    B: ₹${Math.round(data.market_context.bus_price).toLocaleString()} (${data.market_context.bus_vs_market}), ${data.market_context.bus_load.toFixed(1)}% full
                </small>
            </div>
        `;
                // Add Q-spread indicator + top-3 alternatives
        const qSpread = data.q_spread || 0;
        const spreadLabel = qSpread > 1.0 ? '🟢 Decisive' : qSpread > 0.3 ? '🟡 Moderate' : '🔴 Uncertain';

        let top3Html = '';
        if (data.top3_actions && data.top3_actions.length > 0) {
            top3Html = `
                <div style="margin-top:12px; border-top:1px solid rgba(255,255,255,0.1); padding-top:10px;">
                    <small style="color:#94a3b8;">Agent considering (Q-spread: ${qSpread.toFixed(2)} ${spreadLabel}):</small>
                    ${data.top3_actions.map((a, i) => `
                        <div style="display:flex; justify-content:space-between; margin-top:5px; 
                                    ${i===0 ? 'color:#10b981;font-weight:bold;' : 'color:#94a3b8;'}">
                            <span>${i===0?'►':' '} ${a.name}</span>
                            <span>Q=${a.q_value.toFixed(2)} (${(a.probability*100).toFixed(0)}%)</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        document.getElementById('ai-rec-content').innerHTML += top3Html;
        
    } catch (error) {
        console.error('Error getting AI recommendation:', error);
        document.getElementById('ai-rec-content').innerHTML = `
            <div class="error">❌ Error loading recommendation</div>
        `;
    }
}

async function loadAgentInfo() {
    try {
        const response = await fetch('/api/agent_info');
        const data = await response.json();
        
        console.log('🤖 RL Agent Info:', data);
        
        const statusText = data.agent_status === 'trained' 
            ? '<span style="color: #10b981;">✓ Trained</span>' 
            : '<span style="color: #f59e0b;">⚠️ Untrained</span>';
        
        document.getElementById('agent-status').innerHTML = statusText;
        document.getElementById('agent-state-size').textContent = data.state_size;
        document.getElementById('agent-action-size').textContent = data.action_size;
        document.getElementById('agent-episodes').textContent = data.episodes_trained || '0';
        
        if (!data.agent_loaded) {
            showToast('⚠️ No trained model loaded - agent using untrained policy', 'warning');
        } else {
            showToast('✓ Trained RL agent loaded successfully', 'success');
        }
    } catch (error) {
        console.error('Error loading agent info:', error);
        document.getElementById('agent-status').innerHTML = '<span style="color: #ef4444;">❌ Error</span>';
    }
}

async function followAI() {
    if (currentAIAction !== null) {
        await takeAction(currentAIAction);
    } else {
        await getAIRecommendation();
        setTimeout(() => {
            if (currentAIAction !== null) {
                takeAction(currentAIAction);
            }
        }, 500);
    }
}

async function autoRun() {
    if (autoRunning) {
        stopAutoRun();
        return;
    }
    
    autoRunning = true;
    const btn = event.target;
    btn.textContent = '⏸️ Stop Auto';
    btn.style.background = '#ef4444';
    
    addLogEntry('🤖 Auto-run started');
    showToast('🤖 Auto-run mode activated', 'info');
    
    autoRunInterval = setInterval(async () => {
        await getAIRecommendation();
        await followAI();
    }, 3000);
}

function stopAutoRun() {
    autoRunning = false;
    clearInterval(autoRunInterval);
    
    const btn = document.querySelector('.btn-auto');
    if (btn) {
        btn.textContent = '🤖 Auto Run (AI Mode)';
        btn.style.background = '';
    }
    
    addLogEntry('⏸️ Auto-run stopped');
}

async function triggerDisruption(type) {
    try {
        const response = await fetch('/api/disruption', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: type })
        });
        
        const data = await response.json();
        
        if (data.success) {
            addLogEntry(data.message);
            showToast(data.message, 'info');
            await updateDashboard();
        }
    } catch (error) {
        console.error('Error triggering disruption:', error);
        showToast('❌ Error triggering disruption', 'error');
    }
}

async function resetSimulation() {
    if (!confirm('Reset multi-class simulation? This will clear all progress.')) return;
    
    stopAutoRun();
    
    try {
        const response = await fetch('/api/reset', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await response.json();
        
        if (data.success) {
            showToast('✅ Multi-class simulation reset', 'success');
            addLogEntry('🔄 Reset to initial state');
            
            sessionStats = {
                actions: 0,
                bookings: 0,
                rewards: []
            };
            
            document.getElementById('actions-count').textContent = '0';
            document.getElementById('total-bookings').textContent = '0';
            document.getElementById('avg-reward').textContent = '0.00';
            
            [econPriceChart, busPriceChart, revenueChart].forEach(chart => {
                chart.data.labels = [];
                chart.data.datasets.forEach(ds => ds.data = []);
                chart.update();
            });
            
            await updateDashboard();
            await getAIRecommendation();
        }
    } catch (error) {
        console.error('Error resetting:', error);
        showToast('❌ Error resetting simulation', 'error');
    }
}

// FIXED: Update history with DAILY revenue (not cumulative)
async function updateHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.history && data.history.length > 0) {
            const last20 = data.history.slice(-20);
            
            const labels = last20.map((h, idx) => {
                const day = h.day || (idx + 1);
                return `Day ${day}`;
            });
            
            // Economy prices
            econPriceChart.data.labels = labels;
            econPriceChart.data.datasets[0].data = last20.map(h => 
                Math.round(h.econ_price || 0)
            );
            econPriceChart.update();
            
            // Business prices
            busPriceChart.data.labels = labels;
            busPriceChart.data.datasets[0].data = last20.map(h => 
                Math.round(h.bus_price || 0)
            );
            busPriceChart.update();
            
            // FIXED: Calculate DAILY revenue from step revenue
            revenueChart.data.labels = labels;
            
            // Each history entry should have the revenue for THAT DAY
            const econDailyRev = last20.map(h => {
                // Get bookings and price for this day
                const econBookings = h.econ_bookings || 0;
                const econPrice = h.econ_price || 0;
                return Math.round(econBookings * econPrice);
            });
            
            const busDailyRev = last20.map(h => {
                const busBookings = h.bus_bookings || 0;
                const busPrice = h.bus_price || 0;
                return Math.round(busBookings * busPrice);
            });
            
            revenueChart.data.datasets[0].data = econDailyRev;
            revenueChart.data.datasets[1].data = busDailyRev;
            revenueChart.update();
        }
    } catch (error) {
        console.error('Error updating history:', error);
    }
}

function addLogEntry(message) {
    const log = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const time = new Date().toLocaleTimeString();
    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-message">${message}</span>
    `;
    
    log.insertBefore(entry, log.firstChild);
    
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}

setTimeout(getAIRecommendation, 1000);

// ============================================================================
// COMPARISON PANEL FUNCTIONALITY
// Add this to the END of your dashboard.js file
// ============================================================================

let comparisonChart = null;
let comparisonData = null;

/**
 * Toggle comparison panel visibility
 */
function toggleComparison() {
    const overlay = document.getElementById('comparison-overlay');
    const panel = document.getElementById('comparison-panel');
    
    overlay.classList.add('active');
    panel.classList.add('active');
    
    // Load cached results if available
    loadCachedComparison();
}

/**
 * Close comparison panel
 */
function closeComparison() {
    const overlay = document.getElementById('comparison-overlay');
    const panel = document.getElementById('comparison-panel');
    
    overlay.classList.remove('active');
    panel.classList.remove('active');
}

/**
 * Run full comparison across all strategies
 */
async function runComparison() {
    const btn = document.getElementById('run-comparison-btn');
    const spinner = document.getElementById('loading-spinner');
    const results = document.getElementById('comparison-results');
    
    btn.disabled = true;
    btn.textContent = '⏳ Running...';
    spinner.classList.add('active');
    results.style.display = 'none';
    
    try {
        const response = await fetch('/api/run_comparison', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ episodes: 10 })
        });
        
        const data = await response.json();
        
        if (data.success) {
            comparisonData = data.results;
            displayComparisonResults(data.results);
            showToast('✅ Comparison complete!', 'success');
        } else {
            showToast('❌ ' + (data.error || 'Comparison failed'), 'error');
        }
    } catch (error) {
        console.error('Error running comparison:', error);
        showToast('❌ Error running comparison', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Run Full Comparison (10 episodes each)';
        spinner.classList.remove('active');
    }
}

/**
 * Load cached comparison results if available
 */
async function loadCachedComparison() {
    try {
        const response = await fetch('/api/get_comparison');
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                comparisonData = data.results;
                displayComparisonResults(data.results);
            }
        }
    } catch (error) {
        // No cached data available - that's fine
        console.log('No cached comparison data');
    }
}

/**
 * Display comparison results in the panel
 */
function displayComparisonResults(results) {
    const resultsContainer = document.getElementById('comparison-results');
    const cardsContainer = document.getElementById('strategy-cards');
    const summaryContainer = document.getElementById('comparison-summary');
    const summaryContent = document.getElementById('summary-content');
    
    resultsContainer.style.display = 'block';
    cardsContainer.innerHTML = '';
    
    // Display summary if RL agent is included
    if (results.comparison_summary) {
        const summary = results.comparison_summary;
        summaryContainer.style.display = 'block';
        
        const improvement = summary.improvement_percent;
        const isPositive = improvement > 0;
        
        summaryContent.innerHTML = `
            <div class="summary-stat">
                <div class="metric-label">RL Agent Revenue</div>
                <div class="metric-value">₹${Math.round(summary.rl_revenue).toLocaleString()}</div>
            </div>
            <div class="summary-stat">
                <div class="metric-label">Best Traditional (${summary.best_traditional.replace('_', ' ')})</div>
                <div class="metric-value">₹${Math.round(summary.best_traditional_revenue).toLocaleString()}</div>
            </div>
            <div class="summary-stat">
                <div class="metric-label">Improvement</div>
                <div class="metric-value">
                    <span class="improvement-badge ${isPositive ? 'positive' : 'negative'}">
                        ${isPositive ? '📈' : '📉'} ${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}%
                    </span>
                </div>
            </div>
        `;
    } else {
        summaryContainer.style.display = 'none';
    }
    
    // Create strategy cards
    const strategies = Object.entries(results).filter(([key]) => key !== 'comparison_summary');
    
    // Sort by revenue (RL first, then by performance)
    strategies.sort((a, b) => {
        if (a[0] === 'rl_agent') return -1;
        if (b[0] === 'rl_agent') return 1;
        return b[1].avg_revenue - a[1].avg_revenue;
    });
    
    strategies.forEach(([strategyKey, metrics]) => {
        const isRL = strategyKey === 'rl_agent';
        const card = document.createElement('div');
        card.className = `strategy-card ${isRL ? 'rl-card' : ''}`;
        
        card.innerHTML = `
            <div class="strategy-name">
                ${isRL ? '🤖' : '📋'} ${metrics.name}
                <span class="strategy-badge ${isRL ? 'rl-badge' : ''}">${isRL ? 'RL AGENT' : 'TRADITIONAL'}</span>
            </div>
            <div class="strategy-metrics">
                <div class="metric-item">
                    <div class="metric-label">Avg Revenue</div>
                    <div class="metric-value">₹${Math.round(metrics.avg_revenue).toLocaleString()}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Load Factor</div>
                    <div class="metric-value">${metrics.avg_load_factor.toFixed(1)}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Economy Load</div>
                    <div class="metric-value">${metrics.avg_econ_load.toFixed(1)}%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Business Load</div>
                    <div class="metric-value">${metrics.avg_bus_load.toFixed(1)}%</div>
                </div>
            </div>
        `;
        
        cardsContainer.appendChild(card);
    });
    
    // Create comparison chart
    createComparisonChart(results);
}

/**
 * Create bar chart comparing all strategies
 */
function createComparisonChart(results) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (comparisonChart) {
        comparisonChart.destroy();
    }
    
    const strategies = Object.entries(results).filter(([key]) => key !== 'comparison_summary');
    const labels = strategies.map(([key, data]) => data.name);
    const revenues = strategies.map(([key, data]) => data.avg_revenue);
    const loadFactors = strategies.map(([key, data]) => data.avg_load_factor);
    
    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Average Revenue (₹)',
                    data: revenues,
                    backgroundColor: strategies.map(([key]) => 
                        key === 'rl_agent' ? 'rgba(16, 185, 129, 0.8)' : 'rgba(102, 126, 234, 0.6)'
                    ),
                    borderColor: strategies.map(([key]) => 
                        key === 'rl_agent' ? '#10b981' : '#667eea'
                    ),
                    borderWidth: 2,
                    yAxisID: 'y',
                },
                {
                    label: 'Load Factor (%)',
                    data: loadFactors,
                    type: 'line',
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 3,
                    pointRadius: 5,
                    pointBackgroundColor: '#f59e0b',
                    yAxisID: 'y1',
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: { 
                    display: true,
                    labels: { color: '#cbd5e1', font: { size: 12 } }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.dataset.yAxisID === 'y') {
                                label += '₹' + context.parsed.y.toLocaleString();
                            } else {
                                label += context.parsed.y.toFixed(1) + '%';
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    beginAtZero: true,
                    ticks: {
                        color: '#cbd5e1',
                        callback: function(value) {
                            return '₹' + (value / 1000).toFixed(0) + 'K';
                        }
                    },
                    grid: { color: 'rgba(203, 213, 225, 0.1)' },
                    title: {
                        display: true,
                        text: 'Revenue (₹)',
                        color: '#cbd5e1'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    min: 0,
                    max: 100,
                    ticks: {
                        color: '#f59e0b',
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
                        color: '#f59e0b'
                    }
                },
                x: {
                    ticks: { 
                        color: '#cbd5e1', 
                        maxRotation: 45, 
                        minRotation: 45 
                    },
                    grid: { color: 'rgba(203, 213, 225, 0.1)' }
                }
            }
        }
    });
}

/**
 * Test a single traditional strategy
 * (Optional - for live testing)
 */
async function testTraditionalStrategy(strategyName) {
    try {
        const response = await fetch('/api/test_traditional', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy: strategyName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(data.message, 'success');
            console.log('Traditional strategy test:', data);
        } else {
            showToast('❌ ' + (data.error || 'Test failed'), 'error');
        }
    } catch (error) {
        console.error('Error testing strategy:', error);
        showToast('❌ Error testing strategy', 'error');
    }
}

// Export functions for external use
window.comparisonFunctions = {
    toggleComparison,
    closeComparison,
    runComparison,
    loadCachedComparison,
    testTraditionalStrategy
};

console.log('✅ Comparison panel functionality loaded');