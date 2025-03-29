
import axios from 'axios';

// Base API URL - use environment variable or default to localhost
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

// Create axios instance with default config
const axiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const api = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await axiosInstance.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Translate hand landmarks
  translateHandLandmarks: async (landmarks: any) => {
    try {
      const response = await axiosInstance.post('/predict', {
        keypoints: landmarks,
      });
      return response.data;
    } catch (error) {
      console.error('Translation request failed:', error);
      throw error;
    }
  },
};
