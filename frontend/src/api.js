/**
 * Central API utility for communicating with the FastAPI backend.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Generic fetch wrapper with error handling.
 */
async function request(endpoint, options = {}) {
  const url = `${API_URL}${endpoint}`;
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Cannot connect to the server. Is the backend running?');
    }
    throw error;
  }
}

/**
 * Health check — GET /health
 */
export async function healthCheck() {
  return request('/health');
}

/**
 * Get feature names — GET /features
 */
export async function getFeatureNames() {
  return request('/features');
}

/**
 * Single prediction — POST /predict
 * @param {number[]} features - Array of 30 feature values
 * @param {string|null} patientLabel - Optional patient label
 */
export async function predictSingle(features, patientLabel = null) {
  return request('/predict', {
    method: 'POST',
    body: JSON.stringify({ features, patient_label: patientLabel }),
  });
}

/**
 * Batch prediction — POST /predict-batch
 * @param {File} csvFile - CSV file with 30 columns per row
 */
export async function predictBatch(csvFile) {
  const formData = new FormData();
  formData.append('file', csvFile);
  return request('/predict-batch', {
    method: 'POST',
    body: formData,
  });
}

/**
 * Get prediction history — GET /history
 */
export async function getHistory() {
  return request('/history');
}

/**
 * Save a prediction record — POST /history
 */
export async function saveHistory(record) {
  return request('/history', {
    method: 'POST',
    body: JSON.stringify(record),
  });
}
