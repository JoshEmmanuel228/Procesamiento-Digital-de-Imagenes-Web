// API endpoint
const API_URL = 'http://localhost:5000';

// DOM Elements
const image1Input = document.getElementById('image1-input');
const image2Input = document.getElementById('image2-input');
const image1 = document.getElementById('image1');
const image2 = document.getElementById('image2');
const resultImage = document.getElementById('result-image');
const downloadBtn = document.getElementById('download-btn');
const parametersPanel = document.getElementById('parameters-panel');
const notification = document.getElementById('notification');
const notificationMessage = document.getElementById('notification-message');
const loadingOverlay = document.getElementById('loading-overlay');

// State
let loadedImages = {
    image1: false,
    image2: false
};
let activeImageNumber = null;

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setupNotification();
    setupDropdowns();

    // Setup histogram modal close button
    document.getElementById('close-histogram-modal').addEventListener('click', () => {
        document.getElementById('histogram-modal').style.display = 'none';
    });

    setupImageInspector();
});

function setupDropdowns() {
    const dropdownBtns = document.querySelectorAll('.dropdown-btn');

    dropdownBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Close other open dropdowns
            document.querySelectorAll('.dropdown-content.show').forEach(openDropdown => {
                if (openDropdown !== btn.nextElementSibling) {
                    openDropdown.classList.remove('show');
                }
            });
            // Toggle current dropdown
            btn.nextElementSibling.classList.toggle('show');
        });
    });

    // Close dropdowns if clicking outside
    window.addEventListener('click', (e) => {
        if (!e.target.matches('.dropdown-btn, .dropdown-btn *')) {
            document.querySelectorAll('.dropdown-content.show').forEach(openDropdown => {
                openDropdown.classList.remove('show');
            });
        }
    });
}

function setupEventListeners() {
    // Image upload handlers
    document.getElementById('image1-input').addEventListener('change', loadImage1);
    document.getElementById('image2-input').addEventListener('change', loadImage2);

    // Operation button handlers
    document.querySelectorAll('.cyber-btn').forEach(btn => {
        btn.addEventListener('click', () => handleOperationClick(btn));
    });

    // Download button handler
    document.getElementById('download-btn').addEventListener('click', handleDownload);

    // Add event listeners for image selection and restoration buttons
    document.querySelectorAll('.image-controls button').forEach(button => {
        button.addEventListener('click', function() {
            const action = this.getAttribute('onclick');
            if (action) {
                // Extract the function name and parameter
                const match = action.match(/(\w+)\((\d+)\)/);
                if (match) {
                    const [_, funcName, param] = match;
                    if (funcName === 'selectImage') {
                        selectImage(parseInt(param));
                    } else if (funcName === 'restoreImage') {
                        restoreImage(parseInt(param));
                    }
                }
            }
        });
    });

    // Add event listeners for arithmetic and logical operations
    document.querySelector('[data-operation="arithmetic"]').addEventListener('click', showArithmeticControls);
    document.querySelector('[data-operation="logical"]').addEventListener('click', showLogicalControls);

    // Add event listeners for connected components operations
    document.querySelectorAll('[data-operation="binary"], [data-operation="vec4"], [data-operation="vec8"], [data-operation="contours"]')
        .forEach(button => {
            button.addEventListener('click', () => {
                const operation = button.dataset.operation;
                handleConnectedComponentsOperation(operation);
            });
        });
}

function setupImageInspector() {
    const viewer = document.getElementById('main-viewer-wrapper');
    const image = document.getElementById('result-image');
    const loupe = document.getElementById('loupe');
    const colorInspector = document.getElementById('color-inspector');
    const colorSwatch = document.getElementById('color-swatch');
    const colorValues = document.getElementById('color-values');

    let zoomLevel = 1;
    let panX = 0;
    let panY = 0;
    let isPanning = false;
    let startPanX = 0;
    let startPanY = 0;

    const updateTransform = () => {
        image.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
    };

    viewer.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        zoomLevel = Math.max(1, zoomLevel + delta); // Prevent zooming out smaller than original
        updateTransform();
    });

    viewer.addEventListener('mousedown', (e) => {
        if (e.button === 0) { // Left mouse button
            isPanning = true;
            viewer.classList.add('panning');
            startPanX = e.clientX - panX;
            startPanY = e.clientY - panY;
        }
    });

    window.addEventListener('mouseup', () => {
        isPanning = false;
        viewer.classList.remove('panning');
    });

    window.addEventListener('mousemove', (e) => {
        if (isPanning) {
            panX = e.clientX - startPanX;
            panY = e.clientY - startPanY;
            updateTransform();
        }
    });

    document.getElementById('zoom-in-btn').addEventListener('click', () => {
        zoomLevel += 0.2;
        updateTransform();
    });

    document.getElementById('zoom-out-btn').addEventListener('click', () => {
        zoomLevel = Math.max(1, zoomLevel - 0.2);
        updateTransform();
    });

    viewer.addEventListener('mousemove', (e) => {
        if (!image.src || image.src.endsWith('/')) return;

        // Create a temporary canvas to get pixel data
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        ctx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);

        const rect = image.getBoundingClientRect();
        // Calculate click coordinates relative to the original image size
        const x = Math.floor((e.clientX - rect.left) / image.clientWidth * image.naturalWidth);
        const y = Math.floor((e.clientY - rect.top) / image.clientHeight * image.naturalHeight);

        const pixelData = ctx.getImageData(x, y, 1, 1).data;
        const [r, g, b] = pixelData;

        // Update Color Inspector
        colorSwatch.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        colorValues.textContent = `RGB: ${r}, ${g}, ${b}`;
        colorValues.style.color = `rgb(${r}, ${g}, ${b})`;
        colorInspector.style.display = 'flex';

        // Update and show Loupe
        const loupeSize = 150;
        const loupeZoom = 4;
        loupe.style.display = 'block';
        loupe.style.left = `${e.clientX - viewer.getBoundingClientRect().left - loupeSize / 2}px`;
        loupe.style.top = `${e.clientY - viewer.getBoundingClientRect().top - loupeSize / 2}px`;
        loupe.style.backgroundImage = `url(${image.src})`;
        loupe.style.backgroundSize = `${image.clientWidth * loupeZoom}px ${image.clientHeight * loupeZoom}px`;

        // --- Corrected Background Position Calculation ---
        // 1. Calculate the ratio of the cursor's position relative to the image element.
        const ratioX = (e.clientX - rect.left) / image.clientWidth;
        const ratioY = (e.clientY - rect.top) / image.clientHeight;

        // 2. Calculate the position of the background image inside the loupe.
        // This is based on the full size of the zoomed background image.
        const bgX = -(ratioX * (image.clientWidth * loupeZoom) - (loupeSize / 2));
        const bgY = -(ratioY * (image.clientHeight * loupeZoom) - (loupeSize / 2));

        const bgPosX = bgX;
        const bgPosY = bgY;
        // --- End of Correction ---

        loupe.style.backgroundPosition = `${bgPosX}px ${bgPosY}px`;
    });

    viewer.addEventListener('mouseleave', () => {
        loupe.style.display = 'none';
        colorInspector.style.display = 'none';
    });
}

function setupNotification() {
    const closeBtn = notification.querySelector('.close-btn');
    closeBtn.addEventListener('click', () => {
        notification.classList.remove('show');
    });
}

// Image Upload Handler
async function handleImageUpload(event, imageType) {
    const file = event.target.files[0];
    if (!file) return;

    showLoading();
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_URL}/load_${imageType}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            const imgElement = document.getElementById(imageType);
            imgElement.src = `data:image/png;base64,${data.image}`;
            activeImageNumber = imageType.replace('image', '');
            showNotification('Imagen cargada exitosamente', 'success');
        } else {
            throw new Error(data.error || 'Fallo al cargar la imagen');
        }
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Operation Click Handler
function handleOperationClick(button) {
    const operation = button.dataset.operation;
    
    // Clear previous parameters
    parametersPanel.innerHTML = '';

    // Make sure the parameters panel container is visible
    document.querySelector('.parameters-panel-container').style.display = 'block';
    
    // Add operation-specific parameters
    switch (operation) {
        case 'grayscale':
            addGrayscaleParameters();
            break;
        case 'threshold':
            addThresholdParameters();
            break;
        case 'noise':
            addNoiseParameters();
            break;
        case 'histogram':
            showHistogramModal();
            break;
        case 'smoothing':
            addSmoothingParameters();
            break;
        case 'edge':
            addEdgeDetectionParameters();
            break;
        case 'morphology':
            addMorphologyParameters();
            break;
        case 'rotation':
            addRotationParameters();
            break;
        case 'scaling':
            addScalingParameters();
            break;
        case 'blend':
            addBlendParameters();
            break;
        case 'watermark':
            addWatermarkParameters();
            break;
        case 'gamma':
            addGammaParameters();
            break;
        case 'color_space':
            showColorSpaceControls(operation);
            break;
        case 'logical':
            showLogicalControls();
            break;
        case 'arithmetic':
            showArithmeticControls();
            break;
        case 'connected_components':
            showConnectedComponentsControls();
            break;
        case 'rgb_channels':
        case 'lab':
        case 'ycrcb':
        case 'hls':
        case 'yuv':
            showColorSpaceControls(operation);
            break;
        case 'linear_filters':
            showLinearFiltersControls();
            break;
        case 'nonlinear_filters':
            showNonLinearFiltersControls();
            break;
        case 'canny':
            showCannyControls();
            break;
        case 'horizontal_edge':
        case 'vertical_edge':
            showDirectionalEdgeControls(operation);
            break;
        case 'sobel':
        case 'prewitt':
        case 'laplacian':
        case 'scharr':
            showBasicEdgeControls(operation);
            break;
        case 'kirsch':
            showKirschControls();
            break;
        case 'roberts':
            showRobertsControls();
            break;
        case 'robinson':
            showRobinsonControls();
            break;
    }
}

// Parameter UI Functions
function addThresholdParameters() {
    const html = `
        <div class="parameter-group">
            <label>Threshold Value</label>
            <input type="range" id="threshold-value" min="0" max="255" value="128">
            <input type="number" id="threshold-value-number" value="128" min="0" max="255">
        </div>
        <div class="parameter-group">
            <label>
                <input type="checkbox" id="threshold-invert">
                Invert
            </label>
        </div>
        <button class="cyber-btn" onclick="applyThreshold()">
            <i class="fas fa-check"></i> Aplicar
        </button>
    `;
    parametersPanel.innerHTML = html;
    setupThresholdListeners();
}

function addNoiseParameters() {
    const html = `
        <div class="parameter-group">
            <label>Noise Type</label>
            <select id="noise-type" class="cyber-select">
                <option value="salt_pepper">Salt & Pepper</option>
                <option value="gaussian">Gaussian</option>
            </select>
        </div>
        <div class="parameter-group" id="salt-pepper-params">
            <label>Amount</label>
            <input type="range" id="noise-amount" min="0" max="1" step="0.1" value="0.5">
            <input type="number" id="noise-amount-number" value="0.5" min="0" max="1" step="0.1">
            <label>Ratio</label>
            <input type="range" id="noise-ratio" min="0" max="1" step="0.1" value="0.5">
            <input type="number" id="noise-ratio-number" value="0.5" min="0" max="1" step="0.1">
        </div>
        <div class="parameter-group" id="gaussian-params" style="display: none;">
            <label>Mean</label>
            <input type="number" id="noise-mean" value="0" min="-255" max="255">
            <label>Standard Deviation</label>
            <input type="number" id="noise-std" value="25" min="0" max="255">
        </div>
        <button class="cyber-btn" onclick="applyNoise()">
            <i class="fas fa-check"></i> Apply
        </button>
    `;
    parametersPanel.innerHTML = html;
    setupNoiseListeners();
}

let histogramChart = null;

async function showHistogramModal() {
    if (!activeImageNumber) {
        showNotification('Por favor, selecciona una imagen primero.', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/get_histogram`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: 'active' })
        });

        if (!response.ok) {
            throw new Error('No se pudo obtener el histograma del servidor.');
        }

        const data = await response.json();
        const histograms = data.histograms;


        if (histogramChart) {
            histogramChart.destroy();
        }

        const datasets = [];
        const labels = Array.from({ length: 256 }, (_, i) => i);

        const createDataset = (label, data, color) => ({
            label: label.charAt(0).toUpperCase() + label.slice(1), // Capitalize label e.g., 'rojo' -> 'Rojo'
            data: data,
            backgroundColor: `rgba(${color}, 0.5)`,
            borderColor: `rgb(${color})`,
            borderWidth: 1,
            hidden: !document.getElementById(`hist-cb-${label.toLowerCase()}`).checked
        });

        if (histograms.red) datasets.push(createDataset('Rojo', histograms.red, '255, 99, 132'));
        if (histograms.green) datasets.push(createDataset('Verde', histograms.green, '75, 192, 192'));
        if (histograms.blue) datasets.push(createDataset('Azul', histograms.blue, '54, 162, 235'));
        if (histograms.luminance) datasets.push(createDataset('Luminancia', histograms.luminance, '201, 203, 207'));

        // Show the modal first
        const modal = document.getElementById('histogram-modal');
        modal.style.display = 'flex';

        const chartConfig = {
            type: 'bar',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#ecf0f1' }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#ecf0f1' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#ecf0f1' } }
                }
            }
        };

        // Render the chart after the modal is visible
        setTimeout(() => {
            const ctx = document.getElementById('histogram-chart').getContext('2d');
            histogramChart = new Chart(ctx, chartConfig);

            const updateChart = () => {
                if (!histogramChart) return;
                histogramChart.config.type = document.getElementById('hist-cb-bars').checked ? 'bar' : 'line';
                histogramChart.options.scales.y.grid.display = document.getElementById('hist-cb-grid').checked;
                histogramChart.options.scales.x.grid.display = document.getElementById('hist-cb-grid').checked;
                
                const yLimit = document.getElementById('hist-y-limit').value;
                histogramChart.options.scales.y.max = yLimit ? parseInt(yLimit) : undefined;

                histogramChart.data.datasets.forEach(ds => {
                    const id = `hist-cb-${ds.label.toLowerCase()}`;
                    const checkbox = document.getElementById(id);
                    if (checkbox) {
                        ds.hidden = !checkbox.checked;
                    }
                });

                histogramChart.update();
            };

            document.querySelectorAll('.histogram-controls input, .histogram-controls select').forEach(el => {
                el.removeEventListener('change', updateChart); // Evita duplicar listeners
                el.addEventListener('change', updateChart);
            });

            document.getElementById('save-histogram-btn').onclick = () => {
                if (!histogramChart) return;
                const link = document.createElement('a');
                link.href = histogramChart.toBase64Image();
                link.download = 'histograma.png';
                link.click();
            };
        }, 100); // A small delay to ensure the modal is rendered

    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// API Call Functions
async function applyThreshold() {
    if (!activeImageNumber || !loadedImages[`image${activeImageNumber}`]) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const value = document.getElementById('threshold-value').value;
    const invert = document.getElementById('threshold-invert').checked;

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_threshold`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                threshold: parseInt(value),
                invert: invert
            })
        });

        const data = await response.json();
        if (response.ok) {
            resultImage.src = `data:image/png;base64,${data.image}`;
            downloadBtn.disabled = false;
            showNotification('Umbral aplicado exitosamente', 'success');
        } else {
            throw new Error(data.error || 'Error al aplicar el umbral');
        }
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyNoise() {
    if (!activeImageNumber || !loadedImages[`image${activeImageNumber}`]) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const type = document.getElementById('noise-type').value;
    const params = type === 'salt_pepper' ? {
        ratio: parseFloat(document.getElementById('noise-ratio').value),
        amount: parseFloat(document.getElementById('noise-amount').value)
    } : {
        mean: parseInt(document.getElementById('noise-mean').value),
        std_dev: parseInt(document.getElementById('noise-std').value)
    };

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_noise`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                type: type,
                ...params
            })
        });

        const data = await response.json();
        if (response.ok) {
            resultImage.src = `data:image/png;base64,${data.image}`;
            downloadBtn.disabled = false;
            showNotification('Ruido aplicado exitosamente', 'success');
        } else {
            throw new Error(data.error || 'Fallo al aplicar ruido');
        }
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Utility Functions
function showNotification(message, type = 'info') {
    notificationMessage.textContent = message;
    notification.className = `notification show ${type}`;
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

function showLoading() {
    loadingOverlay.classList.add('show');
}

function hideLoading() {
    loadingOverlay.classList.remove('show');
}

function handleDownload() {
    if (!resultImage.src) return;
    
    const link = document.createElement('a');
    link.href = resultImage.src;
    link.download = 'processed_image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Event Listener Setup Functions
function setupThresholdListeners() {
    const slider = document.getElementById('threshold-value');
    const number = document.getElementById('threshold-value-number');
    
    slider.addEventListener('input', (e) => {
        number.value = e.target.value;
    });
    
    number.addEventListener('input', (e) => {
        slider.value = e.target.value;
    });
}

function setupNoiseListeners() {
    const typeSelect = document.getElementById('noise-type');
    const saltPepperParams = document.getElementById('salt-pepper-params');
    const gaussianParams = document.getElementById('gaussian-params');
    
    typeSelect.addEventListener('change', (e) => {
        if (e.target.value === 'salt_pepper') {
            saltPepperParams.style.display = 'block';
            gaussianParams.style.display = 'none';
        } else {
            saltPepperParams.style.display = 'none';
            gaussianParams.style.display = 'block';
        }
    });
    
    // Setup range/number input pairs
    ['amount', 'ratio'].forEach(param => {
        const slider = document.getElementById(`noise-${param}`);
        const number = document.getElementById(`noise-${param}-number`);
        
        slider.addEventListener('input', (e) => {
            number.value = e.target.value;
        });
        
        number.addEventListener('input', (e => {
            slider.value = e.target.value;
        }));
    });
}

// Add debounce function at the top level
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add these functions for image selection and restoration
async function selectImage(imageNumber) {
    try {
        if (!loadedImages[`image${imageNumber}`]) {
            showNotification(`No hay imagen ${imageNumber} cargada`, 'error');
            return;
        }

        const response = await fetch(`${API_URL}/select_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_number: imageNumber })
        });
        
        const data = await response.json();
        if (response.ok) {
            document.getElementById('result-image').src = `data:image/png;base64,${data.image}`;
            activeImageNumber = imageNumber;
            showNotification(`Imagen ${imageNumber} seleccionada`, 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        showNotification('Error al seleccionar la imagen', 'error');
    }
}

async function restoreImage(imageNumber) {
    try {
        if (!loadedImages[`image${imageNumber}`]) {
            showNotification(`No hay imagen ${imageNumber} cargada`, 'error');
            return;
        }

        const response = await fetch(`${API_URL}/restore_image`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image_number: imageNumber })
        });
        
        const data = await response.json();
        if (response.ok) {
            document.getElementById('result-image').src = `data:image/png;base64,${data.image}`;
            activeImageNumber = imageNumber;
            showNotification(`Imagen ${imageNumber} restaurada`, 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        showNotification('Error al restaurar la imagen', 'error');
    }
}

// Update the grayscale parameters function
function addGrayscaleParameters() {
    const html = `
        <div class="parameter-group">
            <label>
                <input type="checkbox" id="invertGray">
                Invertir Escala de Grises
            </label>
        </div>
        <button class="cyber-btn" onclick="applyGrayscale()">
            <i class="fas fa-check"></i> Aplicar
        </button>
    `;
    parametersPanel.innerHTML = html;

    // Add event listener for the checkbox
    document.getElementById('invertGray').addEventListener('change', applyGrayscale);
}

// Update the grayscale function
async function applyGrayscale() {
    try {
        if (!activeImageNumber || !loadedImages[`image${activeImageNumber}`]) {
            showNotification('Por favor selecciona una imagen primero', 'error');
            return;
        }

        showLoading();
        const invert = document.getElementById('invertGray').checked;
        
        const response = await fetch(`${API_URL}/apply_gray`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                invert: invert,
                image_number: activeImageNumber 
            })
        });
        
        const data = await response.json();
        if (response.ok) {
            resultImage.src = `data:image/png;base64,${data.image}`;
            showNotification('Escala de grises aplicada', 'success');
        } else {
            showNotification(data.error || 'Error al aplicar escala de grises', 'error');
        }
    } catch (error) {
        showNotification('Error al aplicar escala de grises', 'error');
    } finally {
        hideLoading();
    }
}

// Update the image loading functions
async function loadImage1() {
    try {
        const fileInput = document.getElementById('image1-input');
        const file = fileInput.files[0];
        if (!file) {
            return;
        }

        showLoading();
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${API_URL}/load_image1`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            document.getElementById('image1').src = `data:image/png;base64,${data.image}`;
            resultImage.src = `data:image/png;base64,${data.image}`;
            loadedImages.image1 = true;
            activeImageNumber = 1;
            showNotification('Imagen 1 cargada', 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        showNotification('Error al cargar la imagen 1', 'error');
    } finally {
        hideLoading();
    }
}

async function loadImage2() {
    try {
        const fileInput = document.getElementById('image2-input');
        const file = fileInput.files[0];
        if (!file) {
            return;
        }

        showLoading();
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${API_URL}/load_image2`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            document.getElementById('image2').src = `data:image/png;base64,${data.image}`;
            resultImage.src = `data:image/png;base64,${data.image}`;
            loadedImages.image2 = true;
            activeImageNumber = 2;
            showNotification('Imagen 2 cargada', 'success');
        } else {
            showNotification(data.error, 'error');
        }
    } catch (error) {
        showNotification('Error al cargar la imagen 2', 'error');
    } finally {
        hideLoading();
    }
}

// Add gamma correction parameters
function addGammaParameters() {
    const html = `
        <div class="parameter-group">
            <label>Valor de Gamma</label>
            <input type="range" id="gamma-value" min="0.1" max="3.0" step="0.1" value="1.0">
            <input type="number" id="gamma-value-number" value="1.0" min="0.1" max="3.0" step="0.1">
        </div>
        <button class="cyber-btn" onclick="applyGamma()">
            <i class="fas fa-check"></i> Aplicar
        </button>
    `;
    parametersPanel.innerHTML = html;

    // Setup range/number input synchronization
    const slider = document.getElementById('gamma-value');
    const number = document.getElementById('gamma-value-number');
    
    slider.addEventListener('input', (e) => {
        number.value = e.target.value;
        applyGamma();
    });
    
    number.addEventListener('input', (e) => {
        slider.value = e.target.value;
        applyGamma();
    });
}

// Add gamma correction function
async function applyGamma() {
    try {
        if (!activeImageNumber || !loadedImages[`image${activeImageNumber}`]) {
            showNotification('Por favor selecciona una imagen primero', 'error');
            return;
        }

        showLoading();
        const gamma = parseFloat(document.getElementById('gamma-value').value);
        
        const response = await fetch(`${API_URL}/apply_gamma`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                gamma: gamma,
                image_number: activeImageNumber 
            })
        });
        
        const data = await response.json();
        if (response.ok) {
            resultImage.src = `data:image/png;base64,${data.image}`;
            showNotification('Corrección de gamma aplicada', 'success');
        } else {
            showNotification(data.error || 'Error al aplicar corrección gamma', 'error');
        }
    } catch (error) {
        showNotification('Error al aplicar corrección gamma', 'error');
    } finally {
        hideLoading();
    }
}

// Color Space Operations
function showColorSpaceControls(operation) {
    hideAllControls();
    
    switch(operation) {
        case 'rgb_channels':
            parametersPanel.innerHTML = `
                <div class="control-group">
                    <h3>Canales RGB</h3>
                    <button onclick="showChannel('R')" class="cyber-button">
                        <i class="fas fa-circle" style="color: #ff0000;"></i> Canal Rojo
                    </button>
                    <button onclick="showChannel('G')" class="cyber-button">
                        <i class="fas fa-circle" style="color: #00ff00;"></i> Canal Verde
                    </button>
                    <button onclick="showChannel('B')" class="cyber-button">
                        <i class="fas fa-circle" style="color: #0000ff;"></i> Canal Azul
                    </button>
                </div>
            `;
            break;
        case 'lab':
        case 'ycrcb':
        case 'hls':
        case 'yuv':
            applyColorSpace(operation.toUpperCase());
            break;
    }
}

async function showChannel(channel) {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/show_channel`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                channel: channel,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al mostrar el canal');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Canal ${channel} mostrado exitosamente`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyColorSpace(space) {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_color_space`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                space: space,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar el espacio de color');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Espacio de color ${space} aplicado exitosamente`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Logical Operations
function showLogicalControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Operaciones Lógicas</h3>
            <button onclick="applyLogicalOperation('AND')" class="cyber-button">
                <i class="fas fa-code"></i> AND (Img 1, Img 2)
            </button>
            <button onclick="applyLogicalOperation('OR')" class="cyber-button">
                <i class="fas fa-code"></i> OR (Img 1, Img 2)
            </button>
            <button onclick="applyLogicalOperation('XOR')" class="cyber-button">
                <i class="fas fa-code"></i> XOR (Img 1, Img 2)
            </button>
            <button onclick="applyLogicalOperation('NOT')" class="cyber-button">
                <i class="fas fa-code"></i> NOT (Imagen Activa)
            </button>
        </div>
    `;
}

// Arithmetic Operations
function showArithmeticControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Operaciones Aritméticas</h3>
            <div class="operation-control">
                <button onclick="applyArithmeticOperation('add')" class="cyber-button">
                    <i class="fas fa-plus"></i> Sumar
                </button>
                <input type="number" id="add-value" value="0" min="-255" max="255">
                <input type="range" id="add-slider" min="-255" max="255" value="0">
            </div>
            <div class="operation-control">
                <button onclick="applyArithmeticOperation('subtract')" class="cyber-button">
                    <i class="fas fa-minus"></i> Restar
                </button>
                <input type="number" id="subtract-value" value="0" min="-255" max="255">
                <input type="range" id="subtract-slider" min="-255" max="255" value="0">
            </div>
            <div class="operation-control">
                <button onclick="applyArithmeticOperation('multiply')" class="cyber-button">
                    <i class="fas fa-times"></i> Multiplicar
                </button>
                <input type="number" id="multiply-value" value="1" min="0" max="10" step="0.1">
            </div>
            <div class="operation-control">
                <button onclick="applyArithmeticOperation('divide')" class="cyber-button">
                    <i class="fas fa-divide"></i> Dividir
                </button>
                <input type="number" id="divide-value" value="1" min="0.1" max="10" step="0.1">
            </div>
        </div>
    `;
    setupArithmeticListeners();
}

// Connected Components Analysis
function showConnectedComponentsControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Etiquetado de Componentes Conexas</h3>
            <button onclick="showBinaryWindow()" class="cyber-button">Mostrar Imagen Binarizada</button>
            <button onclick="showVec4Window()" class="cyber-button">Mostrar Etiquetas Vecindad-4</button>
            <button onclick="showVec8Window()" class="cyber-button">Mostrar Etiquetas Vecindad-8</button>
            <button onclick="showContoursWindow()" class="cyber-button">Mostrar Contornos y Numeración</button>
        </div>
    `;
}

// Helper function to hide all controls
function hideAllControls() {
    const controlGroups = parametersPanel.getElementsByClassName('control-group');
    for (let group of controlGroups) {
        group.style.display = 'none';
    }
}

// Setup arithmetic operation listeners
function setupArithmeticListeners() {
    // Add slider listeners
    const addSlider = document.getElementById('add-slider');
    const addValue = document.getElementById('add-value');
    const subtractSlider = document.getElementById('subtract-slider');
    const subtractValue = document.getElementById('subtract-value');
    const multiplyValue = document.getElementById('multiply-value');
    const divideValue = document.getElementById('divide-value');

    if (addSlider && addValue) {
        addSlider.addEventListener('input', function() {
            addValue.value = this.value;
            applyArithmeticOperation('add');
        });
        addValue.addEventListener('input', function() {
            addSlider.value = this.value;
            applyArithmeticOperation('add');
        });
    }

    if (subtractSlider && subtractValue) {
        subtractSlider.addEventListener('input', function() {
            subtractValue.value = this.value;
            applyArithmeticOperation('subtract');
        });
        subtractValue.addEventListener('input', function() {
            subtractSlider.value = this.value;
            applyArithmeticOperation('subtract');
        });
    }

    if (multiplyValue) {
        multiplyValue.addEventListener('input', function() {
            applyArithmeticOperation('multiply');
        });
    }

    if (divideValue) {
        divideValue.addEventListener('input', function() {
            applyArithmeticOperation('divide');
        });
    }
}

// Logical Operations
async function applyLogicalOperation(operation) {
    if (!loadedImages.image1 || !loadedImages.image2) {
        showNotification('Se necesitan ambas imágenes para operaciones lógicas', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_logical`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                operation: operation
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar la operación lógica');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Operación lógica ${operation} aplicada`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Arithmetic Operations
async function applyArithmeticOperation(operation) {
    if (!activeImageNumber) {
        showNotification('Selecciona una imagen primero', 'error');
        return;
    }

    const value = document.getElementById(`${operation}-value`).value;
    if (value === '') {
        showNotification('Ingresa un valor válido', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_arithmetic`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                operation: operation,
                value: parseFloat(value),
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar la operación aritmética');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Operación aritmética aplicada`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Connected Components Analysis
async function showBinaryWindow() {
    if (!activeImageNumber) {
        showNotification('Selecciona una imagen primero', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/show_binary`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al mostrar la imagen binarizada');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification('Imagen binarizada mostrada');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function showVec4Window() {
    if (!activeImageNumber) {
        showNotification('Selecciona una imagen primero', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/show_vec4`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al mostrar etiquetas vecindad-4');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification('Etiquetas de vecindad 4 mostradas');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function showVec8Window() {
    if (!activeImageNumber) {
        showNotification('Selecciona una imagen primero', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/show_vec8`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al mostrar etiquetas vecindad-8');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification('Etiquetas de vecindad 8 mostradas');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function showContoursWindow() {
    if (!activeImageNumber) {
        showNotification('Selecciona una imagen primero', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/show_contours`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al mostrar contornos');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification('Contornos mostrados');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

function handleConnectedComponentsOperation(operation) {
    if (!activeImageNumber) {
        showNotification('Por favor carga una imagen primero', 'error');
        return;
    }

    const imageNumber = activeImageNumber === 1 ? 1 : 2;
    const endpoint = `/show_${operation}`;

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image_number: imageNumber })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showNotification(data.error, 'error');
            return;
        }
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(data.message, 'success');
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error al procesar la imagen', 'error');
    });
}

// Linear Filters Controls
function showLinearFiltersControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Filtros Lineales (Pasa-Bajas)</h3>
            <div class="filter-controls">
                <button onclick="applyLinearFilter('average')" class="cyber-button">
                    <i class="fas fa-equals"></i> Filtro de Promedio
                </button>
                <div class="parameter-group">
                    <label>Tamaño del Kernel</label>
                    <input type="range" id="average-kernel" min="3" max="15" step="2" value="3">
                    <input type="number" id="average-kernel-number" value="3" min="3" max="15" step="2">
                </div>
            </div>
            <div class="filter-controls">
                <button onclick="applyLinearFilter('weighted')" class="cyber-button">
                    <i class="fas fa-balance-scale"></i> Filtro de Promedio Ponderado
                </button>
                <div class="parameter-group">
                    <label>Tamaño del Kernel</label>
                    <input type="range" id="weighted-kernel" min="3" max="15" step="2" value="3">
                    <input type="number" id="weighted-kernel-number" value="3" min="3" max="15" step="2">
                </div>
            </div>
            <div class="filter-controls">
                <button onclick="applyLinearFilter('gaussian')" class="cyber-button">
                    <i class="fas fa-bell"></i> Filtro Gaussiano
                </button>
                <div class="parameter-group">
                    <label>Tamaño del Kernel</label>
                    <input type="range" id="gaussian-kernel" min="3" max="15" step="2" value="3">
                    <input type="number" id="gaussian-kernel-number" value="3" min="3" max="15" step="2">
                    <label>Sigma</label>
                    <input type="range" id="gaussian-sigma" min="0.1" max="5" step="0.1" value="1">
                    <input type="number" id="gaussian-sigma-number" value="1" min="0.1" max="5" step="0.1">
                </div>
            </div>
        </div>
    `;
    setupLinearFilterListeners();
}

// Non-linear Filters Controls
function showNonLinearFiltersControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Filtros No Lineales (de Orden)</h3>
            <div class="filter-controls">
                <button onclick="applyNonLinearFilter('median')" class="cyber-button">
                    <i class="fas fa-sort-numeric-up"></i> Filtro de Mediana
                </button>
                <div class="parameter-group">
                    <label>Tamaño del Kernel</label>
                    <input type="range" id="median-kernel" min="3" max="15" step="2" value="3">
                    <input type="number" id="median-kernel-number" value="3" min="3" max="15" step="2">
                </div>
            </div>
            <div class="filter-controls">
                <button onclick="applyNonLinearFilter('mode')" class="cyber-button">
                    <i class="fas fa-chart-bar"></i> Filtro de Moda
                </button>
                <div class="parameter-group">
                    <label>Tamaño del Kernel</label>
                    <input type="range" id="mode-kernel" min="3" max="15" step="2" value="3">
                    <input type="number" id="mode-kernel-number" value="3" min="3" max="15" step="2">
                </div>
            </div>
            <div class="filter-controls">
                <button onclick="applyNonLinearFilter('max')" class="cyber-button">
                    <i class="fas fa-arrow-up"></i> Filtro de Máximo
                </button>
                <div class="parameter-group">
                    <label>Tamaño del Kernel</label>
                    <input type="range" id="max-kernel" min="3" max="15" step="2" value="3">
                    <input type="number" id="max-kernel-number" value="3" min="3" max="15" step="2">
                </div>
            </div>
            <div class="filter-controls">
                <button onclick="applyNonLinearFilter('min')" class="cyber-button">
                    <i class="fas fa-arrow-down"></i> Filtro de Mínimo
                </button>
                <div class="parameter-group">
                    <label>Tamaño del Kernel</label>
                    <input type="range" id="min-kernel" min="3" max="15" step="2" value="3">
                    <input type="number" id="min-kernel-number" value="3" min="3" max="15" step="2">
                </div>
            </div>
        </div>
    `;
    setupNonLinearFilterListeners();
}

// Setup listeners for linear filters
function setupLinearFilterListeners() {
    // Average filter listeners
    const averageKernel = document.getElementById('average-kernel');
    const averageKernelNumber = document.getElementById('average-kernel-number');
    if (averageKernel && averageKernelNumber) {
        averageKernel.addEventListener('input', () => {
            averageKernelNumber.value = averageKernel.value;
        });
        averageKernelNumber.addEventListener('input', () => {
            averageKernel.value = averageKernelNumber.value;
        });
    }

    // Weighted filter listeners
    const weightedKernel = document.getElementById('weighted-kernel');
    const weightedKernelNumber = document.getElementById('weighted-kernel-number');
    if (weightedKernel && weightedKernelNumber) {
        weightedKernel.addEventListener('input', () => {
            weightedKernelNumber.value = weightedKernel.value;
        });
        weightedKernelNumber.addEventListener('input', () => {
            weightedKernel.value = weightedKernelNumber.value;
        });
    }

    // Gaussian filter listeners
    const gaussianKernel = document.getElementById('gaussian-kernel');
    const gaussianKernelNumber = document.getElementById('gaussian-kernel-number');
    const gaussianSigma = document.getElementById('gaussian-sigma');
    const gaussianSigmaNumber = document.getElementById('gaussian-sigma-number');

    if (gaussianKernel && gaussianKernelNumber) {
        gaussianKernel.addEventListener('input', () => {
            gaussianKernelNumber.value = gaussianKernel.value;
        });
        gaussianKernelNumber.addEventListener('input', () => {
            gaussianKernel.value = gaussianKernelNumber.value;
        });
    }

    if (gaussianSigma && gaussianSigmaNumber) {
        gaussianSigma.addEventListener('input', () => {
            gaussianSigmaNumber.value = gaussianSigma.value;
        });
        gaussianSigmaNumber.addEventListener('input', () => {
            gaussianSigma.value = gaussianSigmaNumber.value;
        });
    }
}

// Setup listeners for non-linear filters
function setupNonLinearFilterListeners() {
    // Median filter listeners
    const medianKernel = document.getElementById('median-kernel');
    const medianKernelNumber = document.getElementById('median-kernel-number');
    if (medianKernel && medianKernelNumber) {
        medianKernel.addEventListener('input', () => {
            medianKernelNumber.value = medianKernel.value;
        });
        medianKernelNumber.addEventListener('input', () => {
            medianKernel.value = medianKernelNumber.value;
        });
    }

    // Mode filter listeners
    const modeKernel = document.getElementById('mode-kernel');
    const modeKernelNumber = document.getElementById('mode-kernel-number');
    if (modeKernel && modeKernelNumber) {
        modeKernel.addEventListener('input', () => {
            modeKernelNumber.value = modeKernel.value;
        });
        modeKernelNumber.addEventListener('input', () => {
            modeKernel.value = modeKernelNumber.value;
        });
    }

    // Max filter listeners
    const maxKernel = document.getElementById('max-kernel');
    const maxKernelNumber = document.getElementById('max-kernel-number');
    if (maxKernel && maxKernelNumber) {
        maxKernel.addEventListener('input', () => {
            maxKernelNumber.value = maxKernel.value;
        });
        maxKernelNumber.addEventListener('input', () => {
            maxKernel.value = maxKernelNumber.value;
        });
    }

    // Min filter listeners
    const minKernel = document.getElementById('min-kernel');
    const minKernelNumber = document.getElementById('min-kernel-number');
    if (minKernel && minKernelNumber) {
        minKernel.addEventListener('input', () => {
            minKernelNumber.value = minKernel.value;
        });
        minKernelNumber.addEventListener('input', () => {
            minKernel.value = minKernelNumber.value;
        });
    }
}

// Apply linear filter
async function applyLinearFilter(filterType) {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const kernelSize = parseInt(document.getElementById(`${filterType}-kernel`).value);
    const params = {
        filter_type: filterType,
        kernel_size: kernelSize,
        image_number: activeImageNumber
    };

    if (filterType === 'gaussian') {
        params.sigma = parseFloat(document.getElementById('gaussian-sigma').value);
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_linear_filter`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(params)
        });

        if (!response.ok) {
            throw new Error('Error al aplicar el filtro lineal');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Filtro ${filterType} aplicado exitosamente`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Apply non-linear filter
async function applyNonLinearFilter(filterType) {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const kernelSize = parseInt(document.getElementById(`${filterType}-kernel`).value);

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_nonlinear_filter`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filter_type: filterType,
                kernel_size: kernelSize,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar el filtro no lineal');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Filtro ${filterType} aplicado exitosamente`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Canny Edge Detection Controls
function showCannyControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Detección de Bordes Canny</h3>
            <div class="parameter-group">
                <label>Umbral 1</label>
                <input type="range" id="canny-threshold1" min="0" max="255" value="100">
                <input type="number" id="canny-threshold1-number" value="100" min="0" max="255">
            </div>
            <div class="parameter-group">
                <label>Umbral 2</label>
                <input type="range" id="canny-threshold2" min="0" max="255" value="200">
                <input type="number" id="canny-threshold2-number" value="200" min="0" max="255">
            </div>
            <button onclick="applyCannyEdge()" class="cyber-button">
                <i class="fas fa-magic"></i> Aplicar Borde Canny
            </button>
        </div>
    `;
    setupCannyListeners();
}

// Directional Edge Controls
function showDirectionalEdgeControls(direction) {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Detección de Borde ${direction === 'horizontal_edge' ? 'Horizontal' : 'Vertical'}</h3>
            <div class="parameter-group">
                <label>Tamaño del Kernel</label>
                <input type="range" id="directional-kernel" min="3" max="15" step="2" value="3">
                <input type="number" id="directional-kernel-number" value="3" min="3" max="15" step="2">
            </div>
            <button onclick="applyDirectionalEdge('${direction}')" class="cyber-button">
                <i class="fas fa-magic"></i> Aplicar Borde ${direction === 'horizontal_edge' ? 'Horizontal' : 'Vertical'}
            </button>
        </div>
    `;
    setupDirectionalEdgeListeners();
}

// Basic Edge Detection Controls
function showBasicEdgeControls(type) {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Detección de Borde ${type.charAt(0).toUpperCase() + type.slice(1)}</h3>
            <div class="parameter-group">
                <label>Tamaño del Kernel</label>
                <input type="range" id="basic-kernel" min="3" max="15" step="2" value="3">
                <input type="number" id="basic-kernel-number" value="3" min="3" max="15" step="2">
            </div>
            <button onclick="applyBasicEdge('${type}')" class="cyber-button">
                <i class="fas fa-magic"></i> Aplicar ${type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
        </div>
    `;
    setupBasicEdgeListeners();
}

// Kirsch Edge Detection Controls
function showKirschControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Detección de Borde Kirsch</h3>
            <div class="parameter-group">
                <label>Umbral</label>
                <input type="range" id="kirsch-threshold" min="0" max="255" value="128">
                <input type="number" id="kirsch-threshold-number" value="128" min="0" max="255">
            </div>
            <button onclick="applyKirschEdge()" class="cyber-button">
                <i class="fas fa-magic"></i> Aplicar Borde Kirsch
            </button>
        </div>
    `;
    setupKirschListeners();
}

// Roberts Edge Detection Controls
function showRobertsControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Detección de Borde Roberts</h3>
            <div class="parameter-group">
                <label>Umbral</label>
                <input type="range" id="roberts-threshold" min="0" max="255" value="128">
                <input type="number" id="roberts-threshold-number" value="128" min="0" max="255">
            </div>
            <button onclick="applyRobertsEdge()" class="cyber-button">
                <i class="fas fa-magic"></i> Aplicar Borde Roberts
            </button>
        </div>
    `;
    setupRobertsListeners();
}

// Robinson Masks Controls
function showRobinsonControls() {
    hideAllControls();
    parametersPanel.innerHTML = `
        <div class="control-group">
            <h3>Máscaras de Robinson</h3>
            <div class="button-grid">
                <button onclick="applyRobinsonMask('complete')" class="cyber-button">
                    <i class="fas fa-compass"></i> Completo
                </button>
                <button onclick="applyRobinsonMask('north')" class="cyber-button">
                    <i class="fas fa-arrow-up"></i> Norte
                </button>
                <button onclick="applyRobinsonMask('south')" class="cyber-button">
                    <i class="fas fa-arrow-down"></i> Sur
                </button>
                <button onclick="applyRobinsonMask('east')" class="cyber-button">
                    <i class="fas fa-arrow-right"></i> Este
                </button>
                <button onclick="applyRobinsonMask('west')" class="cyber-button">
                    <i class="fas fa-arrow-left"></i> Oeste
                </button>
                <button onclick="applyRobinsonMask('northwest')" class="cyber-button">
                    <i class="fas fa-arrow-up-left"></i> Noroeste
                </button>
                <button onclick="applyRobinsonMask('southwest')" class="cyber-button">
                    <i class="fas fa-arrow-down-left"></i> Suroeste
                </button>
                <button onclick="applyRobinsonMask('northeast')" class="cyber-button">
                    <i class="fas fa-arrow-up-right"></i> Noreste
                </button>
                <button onclick="applyRobinsonMask('southeast')" class="cyber-button">
                    <i class="fas fa-arrow-down-right"></i> Sureste
                </button>
            </div>
        </div>
    `;
}

// Setup listeners for edge detection controls
function setupCannyListeners() {
    const threshold1 = document.getElementById('canny-threshold1');
    const threshold1Number = document.getElementById('canny-threshold1-number');
    const threshold2 = document.getElementById('canny-threshold2');
    const threshold2Number = document.getElementById('canny-threshold2-number');

    if (threshold1 && threshold1Number) {
        threshold1.addEventListener('input', () => {
            threshold1Number.value = threshold1.value;
        });
        threshold1Number.addEventListener('input', () => {
            threshold1.value = threshold1Number.value;
        });
    }

    if (threshold2 && threshold2Number) {
        threshold2.addEventListener('input', () => {
            threshold2Number.value = threshold2.value;
        });
        threshold2Number.addEventListener('input', () => {
            threshold2.value = threshold2Number.value;
        });
    }
}

function setupDirectionalEdgeListeners() {
    const kernel = document.getElementById('directional-kernel');
    const kernelNumber = document.getElementById('directional-kernel-number');

    if (kernel && kernelNumber) {
        kernel.addEventListener('input', () => {
            kernelNumber.value = kernel.value;
        });
        kernelNumber.addEventListener('input', () => {
            kernel.value = kernelNumber.value;
        });
    }
}

function setupBasicEdgeListeners() {
    const kernel = document.getElementById('basic-kernel');
    const kernelNumber = document.getElementById('basic-kernel-number');

    if (kernel && kernelNumber) {
        kernel.addEventListener('input', () => {
            kernelNumber.value = kernel.value;
        });
        kernelNumber.addEventListener('input', () => {
            kernel.value = kernelNumber.value;
        });
    }
}

function setupKirschListeners() {
    const threshold = document.getElementById('kirsch-threshold');
    const thresholdNumber = document.getElementById('kirsch-threshold-number');

    if (threshold && thresholdNumber) {
        threshold.addEventListener('input', () => {
            thresholdNumber.value = threshold.value;
        });
        thresholdNumber.addEventListener('input', () => {
            threshold.value = thresholdNumber.value;
        });
    }
}

function setupRobertsListeners() {
    const threshold = document.getElementById('roberts-threshold');
    const thresholdNumber = document.getElementById('roberts-threshold-number');

    if (threshold && thresholdNumber) {
        threshold.addEventListener('input', () => {
            thresholdNumber.value = threshold.value;
        });
        thresholdNumber.addEventListener('input', () => {
            threshold.value = thresholdNumber.value;
        });
    }
}

// Edge detection application functions
async function applyCannyEdge() {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const threshold1 = parseInt(document.getElementById('canny-threshold1').value);
    const threshold2 = parseInt(document.getElementById('canny-threshold2').value);

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_edge_detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                method: 'canny',
                threshold1: threshold1,
                threshold2: threshold2,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar la detección de bordes Canny');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification('Detección de bordes Canny aplicada exitosamente');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyDirectionalEdge(direction) {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const kernelSize = parseInt(document.getElementById('directional-kernel').value);

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_edge_detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                method: direction,
                kernel_size: kernelSize,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error(`Error al aplicar la detección de bordes ${direction}`);
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Detección de bordes ${direction} aplicada exitosamente`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyBasicEdge(type) {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const kernelSize = parseInt(document.getElementById('basic-kernel').value);

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_edge_detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                method: type,
                kernel_size: kernelSize,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error(`Error al aplicar la detección de bordes ${type}`);
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Detección de bordes ${type} aplicada exitosamente`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyKirschEdge() {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const threshold = parseInt(document.getElementById('kirsch-threshold').value);

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_edge_detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                method: 'kirsch',
                threshold: threshold,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar la detección de bordes Kirsch');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification('Detección de bordes Kirsch aplicada exitosamente');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyRobertsEdge() {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    const threshold = parseInt(document.getElementById('roberts-threshold').value);

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_edge_detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                method: 'roberts',
                threshold: threshold,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar la detección de bordes Roberts');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification('Detección de bordes Roberts aplicada exitosamente');
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function applyRobinsonMask(direction) {
    if (!activeImageNumber) {
        showNotification('Por favor selecciona una imagen primero', 'error');
        return;
    }

    showLoading();
    try {
        const response = await fetch(`${API_URL}/apply_edge_detection`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                method: 'robinson',
                direction: direction,
                image_number: activeImageNumber
            })
        });

        if (!response.ok) {
            throw new Error('Error al aplicar la máscara de Robinson');
        }

        const data = await response.json();
        resultImage.src = `data:image/png;base64,${data.image}`;
        showNotification(`Máscara de Robinson ${direction} aplicada exitosamente`);
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        hideLoading();
    }
} 