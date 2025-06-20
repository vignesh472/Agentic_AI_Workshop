import axios from 'axios';

const API_URL = 'http://localhost:5000/api/evaluate/mastery';

export const evaluateMastery = async (formData) => {
  try {
    const response = await axios.post(API_URL, {
      learner_id: formData.learner_id,
      quiz_scores: formData.quiz_scores,
      coding_logs: formData.coding_logs,
      retry_data: formData.retry_data,
      time_data: formData.time_data
    }, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 15000 // 15 seconds timeout for AI processing
    });

    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.message || 'Evaluation failed');
    } else if (error.request) {
      throw new Error('No response from server. Please try again.');
    } else {
      throw new Error('Request failed. Please check your connection.');
    }
  }
};