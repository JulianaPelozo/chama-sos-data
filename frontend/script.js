const API_BASE_URL = 'http://localhost:5000/api';

window.addEventListener('load', function() {
    setTimeout(function() {
        const loading = document.getElementById('loading-screen');
        if (loading) {
            loading.style.transition = 'opacity 0.5s';
            loading.style.opacity = '0';
            setTimeout(() => loading.style.display = 'none', 500);
        }
        
        fetch('http://localhost:5000/api/dashboard-stats')
            .catch(() => {
                alert('Backend não encontrado em localhost:5000\n\n' +
                      '1. Verifique se o Flask está rodando\n' +
                      '2. Execute: python backend/app.py\n' +
                      '3. Recarregue esta página');
            });
    }, 2000);
});

// Elementos do DOM
let currentSection = 'dashboard';
let occurrencesData = [];
let currentPage = 1;
let rowsPerPage = 10;
let searchTerm = '';

// Inicialização
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    loadDashboardData();
});
function initializeApp() {
    updateConnectionStatus('connecting');
    
    checkBackendConnection();
    
    setTimeout(() => {
        hideLoadingScreen();
        const statusDot = document.querySelector('.status-dot');
        if (!statusDot.classList.contains('connected')) {
            updateConnectionStatus(false);
            showWarning('Backend não está respondendo. Algumas funcionalidades podem não estar disponíveis.');
        }
    }, 3000);
    
    // Ativar seção inicial
    activateSection('dashboard');
}

// ===== FUNÇÕES AUXILIARES =====
function hideLoadingScreen() {
    const loadingScreen = document.getElementById('loading-screen');
    loadingScreen.style.opacity = '0';
    loadingScreen.style.transition = 'opacity 0.5s ease';
    
    setTimeout(() => {
        loadingScreen.style.display = 'none';
    }, 500);
}

function updateConnectionStatus(status) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-indicator span:last-child');
    
    if (status === 'connecting') {
        statusDot.className = 'status-dot connecting';
        statusText.textContent = 'Conectando...';
    } else if (status === true) {
        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Conectado';
    } else {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Desconectado';
    }
}

function showWarning(message) {
    // Criar notificação de aviso
    const warningDiv = document.createElement('div');
    warningDiv.className = 'warning-notification';
    warningDiv.innerHTML = `
        <div class="warning-content">
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
            <button class="warning-close">&times;</button>
        </div>
    `;
    
    document.body.appendChild(warningDiv);
    
    // Remover após 10 segundos
    setTimeout(() => {
        warningDiv.remove();
    }, 10000);
    
    // Botão para fechar
    warningDiv.querySelector('.warning-close').addEventListener('click', () => {
        warningDiv.remove();
    });
}

// ===== EVENT LISTENERS =====
function setupEventListeners() {
    // Navegação
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('href').substring(1);
            activateSection(section);
        });
    });
    
    document.getElementById('refresh-btn').addEventListener('click', refreshAllData);
    
    document.querySelectorAll('.time-filter').forEach(btn => {
        btn.addEventListener('click', function() {
            document.querySelectorAll('.time-filter').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            updateResponseTimeChart(this.dataset.period);
        });
    });
    
    document.getElementById('prediction-form').addEventListener('submit', handlePredictionSubmit);
    document.getElementById('clear-form').addEventListener('click', clearPredictionForm);
    document.getElementById('close-result').addEventListener('click', closePredictionResult);
    document.getElementById('save-prediction').addEventListener('click', savePredictionAsOccurrence);
    document.getElementById('new-prediction').addEventListener('click', clearPredictionForm);
    
    document.getElementById('retrain-model').addEventListener('click', retrainModel);
    document.getElementById('view-features').addEventListener('click', showFeatureDetails);
    
    document.getElementById('new-occurrence-form').addEventListener('submit', handleNewOccurrenceSubmit);
    
    document.getElementById('search-occurrences').addEventListener('input', handleSearch);
    document.getElementById('rows-per-page').addEventListener('change', handleRowsPerPageChange);
    document.getElementById('prev-page').addEventListener('click', goToPrevPage);
    document.getElementById('next-page').addEventListener('click', goToNextPage);
    
    document.getElementById('time-series-metric').addEventListener('change', updateTimeSeriesChart);
}

function activateSection(sectionId) {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${sectionId}`) {
            link.classList.add('active');
        }
    });
    
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
        if (section.id === sectionId) {
            section.classList.add('active');
        }
    });
    
    currentSection = sectionId;
    
    switch(sectionId) {
        case 'dashboard':
            loadDashboardData();
            break;
        case 'prediction':
            loadModelInfo();
            loadFeatureImportance();
            break;
        case 'occurrences':
            loadOccurrences();
            break;
        case 'analytics':
            loadAnalyticsData();
            break;
    }
}

function checkBackendConnection() {
    fetch(`${API_BASE_URL}/dashboard-stats`)
        .then(response => {
            if (response.ok) {
                updateConnectionStatus(true);
            } else {
                updateConnectionStatus(false);
            }
        })
        .catch(error => {
            console.error('Erro na conexão com o backend:', error);
            updateConnectionStatus(false);
        });
}

function updateConnectionStatus(connected) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-indicator span:last-child');
    
    if (connected) {
        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Conectado';
    } else {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Desconectado';
    }
}

function loadDashboardData() {
    fetch(`${API_BASE_URL}/dashboard-stats`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateStatsCards(data.stats);
                updateGroupChart(data.distributions.grupo);
                updatePriorityChart(data.distributions.prioridade);
                updateResponseTimeChart('month');
            }
        })
        .catch(error => {
            console.error('Erro ao carregar dados do dashboard:', error);
            showError('Erro ao carregar dados do dashboard');
        });
}

function updateStatsCards(stats) {
    const statsGrid = document.getElementById('stats-cards');
    statsGrid.innerHTML = `
        <div class="stat-card total">
            <div class="stat-header">
                <div class="stat-icon total">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="stat-change positive">
                    <i class="fas fa-arrow-up"></i> 5.2%
                </div>
            </div>
            <div class="stat-value">${stats.total_ocorrencias.toLocaleString()}</div>
            <div class="stat-label">Total de Ocorrências</div>
        </div>
        
        <div class="stat-card time">
            <div class="stat-header">
                <div class="stat-icon time">
                    <i class="fas fa-clock"></i>
                </div>
                <div class="stat-change negative">
                    <i class="fas fa-arrow-down"></i> 2.1%
                </div>
            </div>
            <div class="stat-value">${stats.tempo_resposta_medio.toFixed(1)}</div>
            <div class="stat-label">Minutos de Resposta</div>
        </div>
        
        <div class="stat-card victims">
            <div class="stat-header">
                <div class="stat-icon victims">
                    <i class="fas fa-user-injured"></i>
                </div>
                <div class="stat-change positive">
                    <i class="fas fa-arrow-up"></i> 8.7%
                </div>
            </div>
            <div class="stat-value">${stats.total_vitimas.toLocaleString()}</div>
            <div class="stat-label">Vítimas Atendidas</div>
        </div>
        
        <div class="stat-card damage">
            <div class="stat-header">
                <div class="stat-icon damage">
                    <i class="fas fa-dollar-sign"></i>
                </div>
                <div class="stat-change positive">
                    <i class="fas fa-arrow-up"></i> 12.3%
                </div>
            </div>
            <div class="stat-value">R$ ${(stats.danos_totais / 1000000).toFixed(1)}M</div>
            <div class="stat-label">Danos Estimados</div>
        </div>
    `;
}

function updateGroupChart(groupData) {
    const groups = Object.keys(groupData);
    const counts = Object.values(groupData);
    
    const trace = {
        x: groups,
        y: counts,
        type: 'bar',
        marker: {
            color: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            line: {
                color: 'white',
                width: 1
            }
        }
    };
    
    const layout = {
        height: 280,
        margin: { t: 10, b: 40, l: 40, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            tickangle: -45
        },
        yaxis: {
            gridcolor: '#e9ecef'
        }
    };
    
    Plotly.newPlot('group-chart', [trace], layout);
}

function updatePriorityChart(priorityData) {
    const labels = Object.keys(priorityData);
    const values = Object.values(priorityData);
    
    const colors = {
        'Crítica': '#DC3545',
        'Alta': '#FFC107',
        'Média': '#17A2B8',
        'Baixa': '#28A745'
    };
    
    const trace = {
        labels: labels,
        values: values,
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: labels.map(label => colors[label]),
            line: {
                color: 'white',
                width: 2
            }
        },
        textinfo: 'label+percent',
        textposition: 'outside',
        hoverinfo: 'label+value+percent'
    };
    
    const layout = {
        height: 280,
        margin: { t: 10, b: 10, l: 10, r: 10 },
        paper_bgcolor: 'transparent',
        showlegend: false
    };
    
    Plotly.newPlot('priority-chart', [trace], layout);
}

function updateResponseTimeChart(period) {
    const sampleData = {
        hour: [15, 18, 12, 20, 16, 14, 22],
        day: [18, 22, 16, 20, 24, 18, 16],
        week: [20, 18, 22, 24, 20, 18, 16, 22, 20, 24, 18, 20, 22, 24],
        month: [22, 24, 20, 18, 22, 24, 20, 18, 16, 20, 22, 24, 20, 18, 22, 24, 20, 18, 16, 22, 20, 24, 18, 20, 22, 24, 20, 18, 22, 24]
    };
    
    const data = sampleData[period];
    const xValues = Array.from({length: data.length}, (_, i) => i + 1);
    
    const trace = {
        x: xValues,
        y: data,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Tempo de Resposta',
        line: {
            color: '#DC3545',
            width: 3
        },
        marker: {
            size: 6,
            color: '#DC3545'
        },
        fill: 'tozeroy',
        fillcolor: 'rgba(220, 53, 69, 0.1)'
    };
    
    const average = data.reduce((a, b) => a + b) / data.length;
    
    const averageTrace = {
        x: [xValues[0], xValues[xValues.length - 1]],
        y: [average, average],
        type: 'scatter',
        mode: 'lines',
        name: `Média: ${average.toFixed(1)} min`,
        line: {
            color: '#6C757D',
            width: 2,
            dash: 'dash'
        }
    };
    
    const layout = {
        height: 300,
        margin: { t: 10, b: 40, l: 50, r: 10 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            title: period === 'hour' ? 'Hora' : period === 'day' ? 'Dia' : period === 'week' ? 'Semana' : 'Dia',
            gridcolor: '#e9ecef'
        },
        yaxis: {
            title: 'Minutos',
            gridcolor: '#e9ecef'
        },
        showlegend: true,
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255, 255, 255, 0.8)'
        }
    };
    
    Plotly.newPlot('response-time-chart', [trace, averageTrace], layout);
}

function loadModelInfo() {
    fetch(`${API_BASE_URL}/model-info`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.metrics) {
                document.getElementById('model-accuracy').textContent = data.metrics.r2.toFixed(3);
                document.getElementById('model-mae').textContent = `${data.metrics.mae.toFixed(1)} min`;
                document.getElementById('last-training').textContent = new Date().toLocaleDateString('pt-BR');
            }
        })
        .catch(error => {
            console.error('Erro ao carregar informações do modelo:', error);
        });
}

function loadFeatureImportance() {
    fetch(`${API_BASE_URL}/model-info`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.feature_importance) {
                updateFeatureImportance(data.feature_importance);
            }
        })
        .catch(error => {
            console.error('Erro ao carregar importância das features:', error);
        });
}

function updateFeatureImportance(features) {
    const container = document.getElementById('feature-importance');
    const sortedFeatures = Object.entries(features).sort((a, b) => b[1] - a[1]);
    
    container.innerHTML = '<h4>Importância das Variáveis:</h4>';
    
    sortedFeatures.forEach(([feature, importance]) => {
        const percentage = (importance * 100).toFixed(1);
        const barWidth = Math.min(percentage * 3, 100);
        
        const featureDiv = document.createElement('div');
        featureDiv.className = 'feature-item';
        featureDiv.innerHTML = `
            <div class="feature-name">${formatFeatureName(feature)}</div>
            <div class="feature-bar-container">
                <div class="feature-bar" style="width: ${barWidth}%"></div>
                <span class="feature-percentage">${percentage}%</span>
            </div>
        `;
        
        container.appendChild(featureDiv);
    });
}

function formatFeatureName(feature) {
    const names = {
        'Grupo de Ocorrência': 'Grupo',
        'Tipo de Ocorrência': 'Tipo',
        'Prioridade': 'Prioridade',
        'Número de Vítimas': 'Vítimas',
        'Danos Estimados (R$)': 'Danos'
    };
    
    return names[feature] || feature;
}

async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const formData = {
        'Grupo de Ocorrência': document.getElementById('grupo').value,
        'Tipo de Ocorrência': document.getElementById('tipo').value,
        'Prioridade': document.getElementById('prioridade').value,
        'Número de Vítimas': parseInt(document.getElementById('vitimas').value),
        'Danos Estimados (R$)': parseInt(document.getElementById('danos').value),
        'Localização': document.getElementById('localizacao').value
    };
    
    // Validação
    if (!formData['Grupo de Ocorrência'] || !formData['Tipo de Ocorrência'] || !formData['Prioridade']) {
        showError('Preencha todos os campos obrigatórios');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showPredictionResult(data.prediction, data.interpretation, formData);
        } else {
            showError(data.message || 'Erro na predição');
        }
    } catch (error) {
        console.error('Erro na predição:', error);
        showError('Erro de conexão com o servidor');
    }
}

function showPredictionResult(prediction, interpretation, formData) {
    const resultDiv = document.getElementById('prediction-result');
    const predictedTime = document.getElementById('predicted-time');
    const predictionClass = document.getElementById('prediction-class');
    const detailsDiv = document.getElementById('occurrence-details');
    
    predictedTime.textContent = prediction.toFixed(1);
    
    let classificationHTML = '';
    let colorClass = '';
    
    if (prediction <= 15) {
        classificationHTML = '<span class="classification-fast"><i class="fas fa-bolt"></i> Resposta RÁPIDA</span>';
        colorClass = 'fast';
    } else if (prediction <= 30) {
        classificationHTML = '<span class="classification-medium"><i class="fas fa-clock"></i> Resposta MÉDIA</span>';
        colorClass = 'medium';
    } else {
        classificationHTML = '<span class="classification-slow"><i class="fas fa-hourglass-half"></i> Resposta LENTA</span>';
        colorClass = 'slow';
    }
    
    predictionClass.innerHTML = classificationHTML;
    predictionClass.className = `prediction-classification ${colorClass}`;
    
    detailsDiv.innerHTML = `
        <div class="detail-item">
            <span class="detail-label">Grupo:</span>
            <span class="detail-value">${formData['Grupo de Ocorrência']}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Tipo:</span>
            <span class="detail-value">${formData['Tipo de Ocorrência']}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Prioridade:</span>
            <span class="detail-value priority-${formData['Prioridade'].toLowerCase()}">${formData['Prioridade']}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Vítimas:</span>
            <span class="detail-value">${formData['Número de Vítimas']}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Danos:</span>
            <span class="detail-value">R$ ${formData['Danos Estimados (R$)'].toLocaleString()}</span>
        </div>
        <div class="detail-item">
            <span class="detail-label">Localização:</span>
            <span class="detail-value">${formData['Localização']}</span>
        </div>
    `;
    
    resultDiv.classList.remove('hidden');
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function clearPredictionForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('vitimas').value = 1;
    document.getElementById('danos').value = 15000;
    document.getElementById('prediction-result').classList.add('hidden');
}

function closePredictionResult() {
    document.getElementById('prediction-result').classList.add('hidden');
}

async function savePredictionAsOccurrence() {
    const formData = {
        'Grupo de Ocorrência': document.getElementById('grupo').value,
        'Tipo de Ocorrência': document.getElementById('tipo').value,
        'Prioridade': document.getElementById('prioridade').value,
        'Número de Vítimas': parseInt(document.getElementById('vitimas').value),
        'Danos Estimados (R$)': parseInt(document.getElementById('danos').value),
        'Localização': document.getElementById('localizacao').value,
        'Tempo de Resposta (minutos)': parseFloat(document.getElementById('predicted-time').textContent)
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/add-occurrence`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccess('Ocorrência salva com sucesso!');
            // Atualizar dashboard e ocorrências
            loadDashboardData();
            loadOccurrences();
        } else {
            showError(data.message || 'Erro ao salvar ocorrência');
        }
    } catch (error) {
        console.error('Erro ao salvar ocorrência:', error);
        showError('Erro de conexão com o servidor');
    }
}

async function retrainModel() {
    if (!confirm('Deseja retreinar o modelo? Isso pode levar alguns minutos.')) {
        return;
    }
    
    showLoading('Treinando modelo...');
    
    try {
        const response = await fetch(`${API_BASE_URL}/train-model`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccess('Modelo treinado com sucesso!');
            loadModelInfo();
            loadFeatureImportance();
        } else {
            showError(data.message || 'Erro ao treinar modelo');
        }
    } catch (error) {
        console.error('Erro ao treinar modelo:', error);
        showError('Erro de conexão com o servidor');
    } finally {
        hideLoading();
    }
}

function showFeatureDetails() {
    // Em produção, pode mostrar um modal com mais detalhes
    alert('Em produção, esta função mostrará detalhes completos das features do modelo.');
}

// ===== OCORRÊNCIAS =====
async function loadOccurrences() {
    try {
        const response = await fetch(`${API_BASE_URL}/recent-occurrences`);
        const data = await response.json();
        
        if (data.success) {
            occurrencesData = data.occurrences;
            renderOccurrencesTable();
        } else {
            showError('Erro ao carregar ocorrências');
        }
    } catch (error) {
        console.error('Erro ao carregar ocorrências:', error);
        showError('Erro de conexão com o servidor');
    }
}

function renderOccurrencesTable() {
    const tbody = document.getElementById('occurrences-table-body');
    const startIndex = (currentPage - 1) * rowsPerPage;
    const endIndex = startIndex + rowsPerPage;
    
    // Filtrar dados se houver busca
    let filteredData = occurrencesData;
    if (searchTerm) {
        const term = searchTerm.toLowerCase();
        filteredData = occurrencesData.filter(occ => 
            occ['Grupo de Ocorrência'].toLowerCase().includes(term) ||
            occ['Tipo de Ocorrência'].toLowerCase().includes(term) ||
            occ['Prioridade'].toLowerCase().includes(term) ||
            occ['Localização'].toLowerCase().includes(term)
        );
    }
    
    const pageData = filteredData.slice(startIndex, endIndex);
    
    // Limpar tabela
    tbody.innerHTML = '';
    
    // Adicionar linhas
    pageData.forEach(occurrence => {
        const row = document.createElement('tr');
        
        // Formatar data
        let dateFormatted = 'N/A';
        if (occurrence.Data) {
            const date = new Date(occurrence.Data);
            dateFormatted = date.toLocaleDateString('pt-BR') + ' ' + date.toLocaleTimeString('pt-BR', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }
        
        const priorityClass = occurrence.Prioridade.toLowerCase();
        
        row.innerHTML = `
            <td>${occurrence.ID || 'N/A'}</td>
            <td>${occurrence['Grupo de Ocorrência'] || 'N/A'}</td>
            <td>${occurrence['Tipo de Ocorrência'] || 'N/A'}</td>
            <td><span class="priority-badge priority-${priorityClass}">${occurrence.Prioridade || 'N/A'}</span></td>
            <td>${occurrence['Tempo de Resposta (minutos)']?.toFixed(1) || 'N/A'} min</td>
            <td>${occurrence['Número de Vítimas'] || '0'}</td>
            <td>R$ ${(occurrence['Danos Estimados (R$)'] || 0).toLocaleString()}</td>
            <td>${dateFormatted}</td>
            <td>
                <button class="btn-icon view-occurrence" data-id="${occurrence.ID}">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="btn-icon delete-occurrence" data-id="${occurrence.ID}">
                    <i class="fas fa-trash"></i>
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
    
    // Atualizar controles de paginação
    updatePaginationControls(filteredData.length);
    
    // Adicionar event listeners aos botões
    document.querySelectorAll('.view-occurrence').forEach(btn => {
        btn.addEventListener('click', function() {
            const id = this.dataset.id;
            viewOccurrenceDetails(id);
        });
    });
    
    document.querySelectorAll('.delete-occurrence').forEach(btn => {
        btn.addEventListener('click', function() {
            const id = this.dataset.id;
            deleteOccurrence(id);
        });
    });
}

function updatePaginationControls(totalItems) {
    const totalPages = Math.ceil(totalItems / rowsPerPage);
    const startItem = ((currentPage - 1) * rowsPerPage) + 1;
    const endItem = Math.min(currentPage * rowsPerPage, totalItems);
    
    // Atualizar informações
    document.getElementById('pagination-info').textContent = 
        `Mostrando ${startItem}-${endItem} de ${totalItems} registros`;
    
    document.getElementById('current-page').textContent = currentPage;
    
    // Atualizar botões
    document.getElementById('prev-page').disabled = currentPage === 1;
    document.getElementById('next-page').disabled = currentPage === totalPages || totalPages === 0;
}

function handleSearch(event) {
    searchTerm = event.target.value.toLowerCase();
    currentPage = 1;
    renderOccurrencesTable();
}

function handleRowsPerPageChange(event) {
    rowsPerPage = parseInt(event.target.value);
    currentPage = 1;
    renderOccurrencesTable();
}

function goToPrevPage() {
    if (currentPage > 1) {
        currentPage--;
        renderOccurrencesTable();
    }
}

function goToNextPage() {
    const totalPages = Math.ceil(occurrencesData.length / rowsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        renderOccurrencesTable();
    }
}

async function handleNewOccurrenceSubmit(event) {
    event.preventDefault();
    
    const formData = {
        'Grupo de Ocorrência': document.getElementById('new-grupo').value,
        'Tipo de Ocorrência': document.getElementById('new-tipo').value,
        'Prioridade': document.getElementById('new-prioridade').value,
        'Tempo de Resposta (minutos)': parseInt(document.getElementById('new-tempo').value),
        'Número de Vítimas': parseInt(document.getElementById('new-vitimas').value),
        'Danos Estimados (R$)': parseInt(document.getElementById('new-danos').value)
    };
    
    // Validação
    if (!formData['Tempo de Resposta (minutos)'] || formData['Tempo de Resposta (minutos)'] < 1) {
        showError('Tempo de resposta deve ser maior que 0');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/add-occurrence`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccess('Ocorrência registrada com sucesso!');
            document.getElementById('new-occurrence-form').reset();
            loadOccurrences();
            loadDashboardData();
        } else {
            showError(data.message || 'Erro ao registrar ocorrência');
        }
    } catch (error) {
        console.error('Erro ao registrar ocorrência:', error);
        showError('Erro de conexão com o servidor');
    }
}

function viewOccurrenceDetails(id) {
    const occurrence = occurrencesData.find(occ => occ.ID == id);
    if (occurrence) {
        const details = `
            ID: ${occurrence.ID || 'N/A'}
            Grupo: ${occurrence['Grupo de Ocorrência'] || 'N/A'}
            Tipo: ${occurrence['Tipo de Ocorrência'] || 'N/A'}
            Prioridade: ${occurrence.Prioridade || 'N/A'}
            Tempo de Resposta: ${occurrence['Tempo de Resposta (minutos)']?.toFixed(1) || 'N/A'} min
            Vítimas: ${occurrence['Número de Vítimas'] || '0'}
            Danos: R$ ${(occurrence['Danos Estimados (R$)'] || 0).toLocaleString()}
            Localização: ${occurrence.Localização || 'N/A'}
            Data: ${occurrence.Data ? new Date(occurrence.Data).toLocaleString('pt-BR') : 'N/A'}
        `;
        alert(details);
    }
}

async function deleteOccurrence(id) {
    if (!confirm('Tem certeza que deseja excluir esta ocorrência?')) {
        return;
    }
    
    try {
        // Nota: Você precisa implementar o endpoint DELETE no backend
        const response = await fetch(`${API_BASE_URL}/delete-occurrence/${id}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccess('Ocorrência excluída com sucesso!');
            loadOccurrences();
            loadDashboardData();
        } else {
            showError(data.message || 'Erro ao excluir ocorrência');
        }
    } catch (error) {
        console.error('Erro ao excluir ocorrência:', error);
        showError('Erro de conexão com o servidor');
    }
}

// ===== ANÁLISES =====
function loadAnalyticsData() {
    updateCorrelationChart();
    updateTimeSeriesChart();
    updateDistributionChart();
    updateSummaryStats();
}

function updateCorrelationChart() {
    // Dados de exemplo para matriz de correlação
    const data = [
        [1.00, 0.65, 0.32, 0.45, 0.21],
        [0.65, 1.00, 0.78, 0.32, 0.54],
        [0.32, 0.78, 1.00, 0.67, 0.43],
        [0.45, 0.32, 0.67, 1.00, 0.89],
        [0.21, 0.54, 0.43, 0.89, 1.00]
    ];
    
    const labels = ['Tempo Resp.', 'Vítimas', 'Danos', 'Prioridade', 'Grupo'];
    
    const trace = {
        z: data,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmid: 0,
        hovertemplate: 'Correlação entre %{x} e %{y}: %{z:.2f}<extra></extra>'
    };
    
    const layout = {
        height: 300,
        margin: { t: 30, b: 40, l: 60, r: 30 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            tickangle: -45
        }
    };
    
    Plotly.newPlot('correlation-chart', [trace], layout);
}

function updateTimeSeriesChart() {
    const metric = document.getElementById('time-series-metric').value;
    
    // Dados de exemplo
    const data = {
        tempo: [18, 22, 16, 20, 24, 18, 16, 20, 22, 24, 20, 18, 16, 22],
        vitimas: [3, 5, 2, 4, 6, 3, 2, 4, 5, 6, 4, 3, 2, 5],
        danos: [25000, 32000, 18000, 28000, 35000, 22000, 19000, 27000, 30000, 38000, 26000, 23000, 20000, 29000]
    };
    
    const labels = {
        tempo: 'Tempo de Resposta (min)',
        vitimas: 'Número de Vítimas',
        danos: 'Danos Estimados (R$)'
    };
    
    const trace = {
        x: Array.from({length: 14}, (_, i) => `Dia ${i + 1}`),
        y: data[metric],
        type: 'scatter',
        mode: 'lines+markers',
        name: labels[metric],
        line: {
            color: '#DC3545',
            width: 3
        },
        marker: {
            size: 6,
            color: '#DC3545'
        }
    };
    
    const layout = {
        height: 300,
        margin: { t: 30, b: 40, l: 60, r: 30 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            gridcolor: '#e9ecef'
        },
        yaxis: {
            title: labels[metric],
            gridcolor: '#e9ecef'
        }
    };
    
    Plotly.newPlot('time-series-chart', [trace], layout);
}

function updateDistributionChart() {
    // Dados de exemplo para distribuição
    const trace = {
        x: [10, 15, 20, 25, 30, 35, 40, 45, 50],
        y: [50, 120, 180, 220, 180, 120, 80, 40, 20],
        type: 'bar',
        name: 'Distribuição',
        marker: {
            color: '#4ECDC4',
            line: {
                color: 'white',
                width: 1
            }
        }
    };
    
    const layout = {
        height: 300,
        margin: { t: 30, b: 40, l: 60, r: 30 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        xaxis: {
            title: 'Tempo de Resposta (min)',
            gridcolor: '#e9ecef'
        },
        yaxis: {
            title: 'Frequência',
            gridcolor: '#e9ecef'
        }
    };
    
    Plotly.newPlot('distribution-chart', [trace], layout);
}

function updateSummaryStats() {
    const container = document.getElementById('summary-stats');
    
    container.innerHTML = `
        <div class="stats-row">
            <div class="stat-item">
                <span class="stat-label">Média de Tempo:</span>
                <span class="stat-value">18.6 min</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Mediana:</span>
                <span class="stat-value">17.2 min</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Moda:</span>
                <span class="stat-value">15 min</span>
            </div>
        </div>
        <div class="stats-row">
            <div class="stat-item">
                <span class="stat-label">Desvio Padrão:</span>
                <span class="stat-value">8.4 min</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Variância:</span>
                <span class="stat-value">70.6</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Coef. Variação:</span>
                <span class="stat-value">45.2%</span>
            </div>
        </div>
        <div class="stats-row">
            <div class="stat-item">
                <span class="stat-label">Mínimo:</span>
                <span class="stat-value">3 min</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Máximo:</span>
                <span class="stat-value">58 min</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Amplitude:</span>
                <span class="stat-value">55 min</span>
            </div>
        </div>
        <div class="stats-row">
            <div class="stat-item">
                <span class="stat-label">Q1 (25%):</span>
                <span class="stat-value">12 min</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Q3 (75%):</span>
                <span class="stat-value">25 min</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">IQR:</span>
                <span class="stat-value">13 min</span>
            </div>
        </div>
    `;
}

// ===== UTILITÁRIOS =====
function refreshAllData() {
    switch(currentSection) {
        case 'dashboard':
            loadDashboardData();
            break;
        case 'prediction':
            loadModelInfo();
            loadFeatureImportance();
            break;
        case 'occurrences':
            loadOccurrences();
            break;
        case 'analytics':
            loadAnalyticsData();
            break;
    }
    
    showSuccess('Dados atualizados!');
}

function showLoading(message) {
    // Implementar um modal de loading se necessário
    console.log('Loading:', message);
}

function hideLoading() {
    console.log('Loading complete');
}

function showSuccess(message) {
    // Implementar notificação toast
    alert('✅ ' + message);
}

function showError(message) {
    // Implementar notificação toast
    alert('❌ ' + message);
}

// Exportar para uso global
window.refreshAllData = refreshAllData;
window.activateSection = activateSection;