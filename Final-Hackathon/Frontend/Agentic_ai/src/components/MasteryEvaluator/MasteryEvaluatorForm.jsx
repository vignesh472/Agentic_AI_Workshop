import { useState } from 'react';
import Loader from '../Loader';
import ErrorMessage from '../ErrorMessage';
import { evaluateMastery } from '../../api/masteryEvaluator';
import AdaptiveAssessModal from './AdaptiveAssessModal';

const initialState = {
  learner_id: '',
  quiz_scores: '',
  coding_logs: '',
  retry_data: '',
  time_data: '',
};

const inputExamples = {
  quiz_scores: "Loops: 85",
  coding_logs: "3 attempts on Loops",
  retry_data: "Loops retried once",
  time_data: "20.5 mins on Loops"
};

const MasteryLevel = ({ mastery }) => {
  const levels = {
    'Strong': { 
      color: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-200',
      icon: '👍'
    },
    'Moderate': { 
      color: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200',
      icon: '👌'
    },
    'Weak': { 
      color: 'bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-200',
      icon: '👎'
    }
  };

  const level = levels[mastery] || levels['Weak'];

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${level.color}`}>
      <span className="mr-1">{level.icon}</span>
      {mastery}
    </span>
  );
};

const ConceptCard = ({ concept }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow border border-gray-200 dark:border-gray-700">
      <div className="flex justify-between items-start">
        <h3 className="text-xl font-bold text-gray-800 dark:text-white">{concept.name}</h3>
        <MasteryLevel mastery={concept.mastery} />
      </div>
      <p className="mt-3 text-gray-600 dark:text-gray-300">{concept.reason}</p>
      <div className="mt-5 pt-4 border-t border-gray-100 dark:border-gray-700">
        <div className="flex items-start">
          <span className="flex-shrink-0 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 p-2 rounded-lg">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </span>
          <p className="ml-3 text-sm text-gray-600 dark:text-gray-300">
            {getRecommendation(concept.mastery, concept.name)}
          </p>
        </div>
      </div>
    </div>
  );
};

const getRecommendation = (mastery, topic) => {
  const recommendations = {
    'Strong': `You're excelling at ${topic}! Challenge yourself with advanced problems.`,
    'Moderate': `You're making progress with ${topic}. Try some intermediate exercises.`,
    'Weak': `Let's strengthen your ${topic} foundation. Start with basic tutorials.`
  };
  return recommendations[mastery] || `Focus on learning ${topic} concepts.`;
};

const ResultSummary = ({ concepts }) => {
  const masteryCounts = concepts.reduce((acc, concept) => {
    acc[concept.mastery] = (acc[concept.mastery] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="bg-gradient-to-r from-purple-50 to-blue-50 dark:from-gray-700 dark:to-gray-800 rounded-2xl p-8 mb-8 border border-gray-200 dark:border-gray-700">
      <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-6">Your Mastery Breakdown</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {Object.entries(masteryCounts).map(([mastery, count]) => (
          <div key={mastery} className={`p-5 rounded-xl border ${
            mastery === 'Strong' ? 'border-emerald-200 dark:border-emerald-900/50 bg-emerald-50/50 dark:bg-emerald-900/20' :
            mastery === 'Moderate' ? 'border-amber-200 dark:border-amber-900/50 bg-amber-50/50 dark:bg-amber-900/20' :
            'border-rose-200 dark:border-rose-900/50 bg-rose-50/50 dark:bg-rose-900/20'
          }`}>
            <div className="flex items-center">
              <MasteryLevel mastery={mastery} />
              <div className="ml-auto text-3xl font-bold text-gray-800 dark:text-white">{count}</div>
            </div>
            <div className="mt-2 text-sm text-gray-500 dark:text-gray-400">concepts</div>
          </div>
        ))}
      </div>
    </div>
  );
};

const PrerequisiteGaps = ({ gapsData }) => {
  if (!gapsData) return null;

  if (gapsData.message) {
    return (
      <div className="mt-8 bg-emerald-50 dark:bg-emerald-900/20 rounded-2xl p-6 flex items-center border border-emerald-200 dark:border-emerald-900/50">
        <span className="flex-shrink-0 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-800 dark:text-emerald-200 p-2 rounded-lg">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </span>
        <span className="ml-4 text-emerald-800 dark:text-emerald-200">{gapsData.message}</span>
      </div>
    );
  }

  return (
    <div className="mt-8 bg-amber-50 dark:bg-amber-900/20 rounded-2xl p-6 border border-amber-200 dark:border-amber-900/50">
      <div className="flex items-center mb-5">
        <span className="flex-shrink-0 bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200 p-2 rounded-lg">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
          </svg>
        </span>
        <h2 className="ml-3 text-xl font-bold text-amber-800 dark:text-amber-200">Knowledge Gaps Identified</h2>
      </div>
      
      <p className="text-amber-700 dark:text-amber-300 mb-6">
        These foundational topics will help strengthen your understanding:
      </p>

      <div className="space-y-6">
        {Object.entries(gapsData).map(([concept, prerequisites], idx) => (
          <div key={idx}>
            <h3 className="font-bold text-gray-800 dark:text-white mb-3">{concept}</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {prerequisites.map((gap, subIdx) => (
                <div key={subIdx} className="flex items-center bg-white dark:bg-gray-800 rounded-lg px-4 py-3 text-sm border border-gray-200 dark:border-gray-700">
                  <span className="flex-shrink-0 bg-amber-100 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200 p-1 rounded mr-3">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </span>
                  <span className="truncate">{gap}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const BoosterResults = ({ boosterResult }) => {
  if (!boosterResult) return null;

  return (
    <div className="mt-8 bg-blue-50 dark:bg-blue-900/20 rounded-2xl p-6 border border-blue-200 dark:border-blue-900/50">
      <div className="flex items-center mb-5">
        <span className="flex-shrink-0 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 p-2 rounded-lg">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.361-6.867 8.21 8.21 0 003 2.48z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 18a3.75 3.75 0 00.495-7.467 5.99 5.99 0 00-1.925 3.546 5.974 5.974 0 01-2.133-1A3.75 3.75 0 0012 18z" />
          </svg>
        </span>
        <h2 className="ml-3 text-xl font-bold text-blue-800 dark:text-blue-200">Personalized Boosters</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {boosterResult.boosters?.map((booster, idx) => (
          <div key={idx} className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700 hover:shadow-md transition-shadow">
            <div className="flex items-center mb-3">
              <span className={`text-xs font-semibold px-3 py-1 rounded-full ${
                booster.booster_type === 'Interactive' ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-200' :
                booster.booster_type === 'Reading' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200' :
                'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200'
              }`}>
                {booster.booster_type.charAt(0).toUpperCase() + booster.booster_type.slice(1)}
              </span>
              <span className="ml-2 text-xs font-semibold px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
                {booster.format.charAt(0).toUpperCase() + booster.format.slice(1)}
              </span>
            </div>
            <h3 className="font-bold text-lg text-gray-800 dark:text-white mb-2">{booster.title}</h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">{booster.description}</p>
            <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {booster.estimated_duration}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const PrerequisiteResults = ({ prereqResult }) => {
  if (!prereqResult) return null;

  return (
    <div className="mt-8 bg-indigo-50 dark:bg-indigo-900/20 rounded-2xl p-6 border border-indigo-200 dark:border-indigo-900/50">
      <div className="flex items-center mb-5">
        <span className="flex-shrink-0 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-800 dark:text-indigo-200 p-2 rounded-lg">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
        </span>
        <h2 className="ml-3 text-xl font-bold text-indigo-800 dark:text-indigo-200">Concept Deep Dive</h2>
      </div>

      {prereqResult.message ? (
        <div className="text-indigo-700 dark:text-indigo-300">{prereqResult.message}</div>
      ) : (
        <div className="space-y-6">
          {prereqResult.retrieved?.map((prereq, idx) => (
            <div key={idx} className="bg-white dark:bg-gray-800 rounded-xl p-5 border border-gray-200 dark:border-gray-700">
              <div className="flex items-center mb-3">
                <h3 className="font-bold text-lg text-gray-800 dark:text-white">{prereq.concept}</h3>
                <span className="ml-auto text-xs font-semibold px-3 py-1 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200">
                  {prereq.source_count} sources
                </span>
              </div>
              <p className="text-gray-600 dark:text-gray-300 mb-4">{prereq.summary}</p>
              <div className="bg-indigo-50 dark:bg-indigo-900/30 rounded-lg p-4">
                <div className="font-semibold text-indigo-700 dark:text-indigo-300 mb-2">Example:</div>
                <p className="text-indigo-800 dark:text-indigo-200">{prereq.example}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default function MasteryEvaluatorForm() {
  const [form, setForm] = useState(initialState);
  const [activeTab, setActiveTab] = useState('quiz');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [GapsData, setGapsData] = useState(null);
  const [adaptiveModalOpen, setAdaptiveModalOpen] = useState(false);
  const [boosterResult, setBoosterResult] = useState(null);
  const [boosterLoading, setBoosterLoading] = useState(false);
  const [prereqResult, setPrereqResult] = useState(null);
  const [prereqLoading, setPrereqLoading] = useState(false);
  const [adaptiveAssessmentResult, setAdaptiveAssessmentResult] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const GapsDatas = result?.results?.gaps_result || null;

const parseAndValidateInputs = (formData) => {
  // Parse each field
  const quizScores = parseInputToDict(formData.quiz_scores, 'quiz_scores');
  const retryData = parseRetryData(formData.retry_data); // Special handling for retry data
  const timeData = parseInputToDict(formData.time_data, 'time_data');

  // Get all unique concepts from all fields
  const allConcepts = [
    ...new Set([
      ...Object.keys(quizScores),
      ...Object.keys(retryData),
      ...Object.keys(timeData)
    ])
  ];

  // Validate concepts match across all fields
  const validationErrors = [];
  
  allConcepts.forEach(concept => {
    if (!quizScores[concept]) {
      validationErrors.push(`Missing quiz score for concept: ${concept}`);
    }
    if (!retryData[concept]) {
      validationErrors.push(`Missing retry data for concept: ${concept}`);
    }
    if (!timeData[concept]) {
      validationErrors.push(`Missing time data for concept: ${concept}`);
    }
  });

  if (validationErrors.length > 0) {
    throw new Error(`Validation failed:\n${validationErrors.join('\n')}`);
  }

  return {
    learner_id: "user123",
    quiz_scores: quizScores,
    coding_logs: {},
    retry_data: retryData, // Now contains full retry strings
    time_data: timeData
  };
};

// Special parser for retry data that preserves full strings
const parseRetryData = (input) => {
  const dict = {};
  if (!input || typeof input !== 'string') return dict;

  const entries = input.split(',').map(entry => entry.trim()).filter(entry => entry);

  entries.forEach(entry => {
    // Extract concept name while preserving the full retry string
    const conceptMatch = entry.match(/^(.+?)(?:\s*retried|\s*\d+\s*times?)/i);
    if (conceptMatch) {
      const concept = conceptMatch[1].trim();
      dict[concept] = entry; // Store the full original string
    } else {
      // Fallback for simpler formats
      const [concept] = entry.split(/\s+/);
      if (concept) {
        dict[concept] = entry;
      }
    }
  });

  return dict;
};

// Original parser for other fields
const parseInputToDict = (input, fieldName) => {
  const dict = {};
  if (!input || typeof input !== 'string') return dict;

  try {
    const entries = input.split(',').map(entry => entry.trim()).filter(entry => entry);

    entries.forEach(entry => {
      let concept, value;

      switch(fieldName) {
        case 'quiz_scores':
          [concept, value] = entry.split(':').map(s => s.trim());
          if (!concept || isNaN(parseFloat(value))) {
            throw new Error(`Invalid quiz score format for: "${entry}"`);
          }
          dict[concept] = parseFloat(value);
          break;

        case 'time_data':
          const timeMatch = entry.match(/^(.+?)?\s*(\d+\.?\d*)\s*(mins?|minutes?|hours?)(?:\s*on\s*(.+))?$/i);
          if (!timeMatch) {
            throw new Error(`Invalid time data format for: "${entry}"`);
          }
          concept = timeMatch[4] || timeMatch[1];
          value = parseFloat(timeMatch[2]);
          const unit = timeMatch[3].toLowerCase();
          if (!concept) {
            throw new Error(`Missing concept in time data: "${entry}"`);
          }
          dict[concept.trim()] = unit.startsWith('h') ? value * 60 : value;
          break;

        default:
          break;
      }
    });
  } catch (error) {
    throw new Error(`Error parsing ${fieldName}: ${error.message}`);
  }

  return dict;
};

const handleSubmit = async (e) => {
  e.preventDefault();
  setLoading(true);
  setError(null);
  setResult(null);
  
  try {
    console.log("Form data before parsing:", form);
    
    const requestData = parseAndValidateInputs(form);
    console.log("Data being sent to API:", requestData);
    
    const res = await evaluateMastery(requestData);
    setResult(res);
  } catch (err) {
    console.error("Validation/parsing error:", err);
    setError(err.message);
  } finally {
    setLoading(false);
  }
};

  const handleFindGap = async () => {
    if (!result) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const weakTopics = result.evaluation.concepts
        .filter(concept => concept.mastery === 'Weak')
        .map(concept => concept.name);

      if (weakTopics.length === 0) {
        setGapsData({ message: "No weak topics found to analyze for gaps" });
        return;
      }

      const response = await fetch('http://localhost:5000/api/evaluate/prerequisite-gaps', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(weakTopics),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze gaps');
      }

      const data = await response.json();
      setGapsData(data);
    } catch (err) {
      setError(err.message || 'Something went wrong analyzing gaps');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (field) => {
    setForm({ ...form, [field]: inputExamples[field] });
  };

  const handleBooster = async () => {
    if (!result) return;
    setBoosterLoading(true);
    setError(null);
    setBoosterResult(null);
    try {
      const concepts = result.results.mastery_result.concepts
        .filter(concept => concept.mastery === 'Weak' || concept.mastery === 'Moderate')
        .map(concept => concept.name);
      const allCategoryConcepts = GapsDatas || {};
      const assessmentResult = adaptiveAssessmentResult;
      
      const response = await fetch('http://localhost:5000/api/evaluate/booster-recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ concepts, preference: 'interactive', category: allCategoryConcepts, assessmentResult }),
      });
      if (!response.ok) {
        throw new Error('Failed to get booster recommendation');
      }
      const data = await response.json();
      setBoosterResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong with booster');
    } finally {
      setBoosterLoading(false);
    }
  };

  const handleGetPrerequisite = async () => {
    if (!result) return;
    setPrereqLoading(true);
    setError(null);
    setPrereqResult(null);

    try {
      const weakTopics = result.results.mastery_result.concepts
        .filter(concept => concept.mastery === 'Weak' || concept.mastery === 'Moderate')
        .map(concept => concept.name);

      const allCategoryConcepts = Object.values(GapsDatas || {}).flat();

      if (weakTopics.length === 0) {
        setPrereqResult({ message: 'No weak topics found to retrieve prerequisites.' });
        setPrereqLoading(false);
        return;
      }

      const response = await fetch('http://localhost:5000/api/evaluate/retrieve-prerequisite', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          concepts: weakTopics,
          category: allCategoryConcepts
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to retrieve prerequisites');
      }

      const data = await response.json();
      setPrereqResult(data);

    } catch (err) {
      setError(err.message || 'Something went wrong retrieving prerequisites');
    } finally {
      setPrereqLoading(false);
    }
  };

  const handleStartNewEvaluation = () => {
    setForm(initialState);
    setResult(null);
    setGapsData(null);
    setBoosterResult(null);
    setPrereqResult(null);
    setAdaptiveAssessmentResult(null);
    setError(null);
    setActiveTab('quiz');
  };

  return (
    <div className="p-8">
      {!result ? (
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-10">
            <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-3">Learning Assessment</h1>
            <p className="text-xl text-gray-600 dark:text-gray-300">
              Enter your learning data to receive personalized insights
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-2xl p-8 shadow-sm border border-gray-200 dark:border-gray-700">
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                  Learner ID
                  <span className="text-rose-500 ml-1">*</span>
                </label>
                <input
                  name="learner_id"
                  value={form.learner_id}
                  onChange={handleChange}
                  required
                  className="w-full px-5 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  placeholder="Enter your unique learner ID"
                />
              </div> */}

              <div className="flex border-b border-gray-200 dark:border-gray-700">
                {[
                  { id: 'quiz', label: 'Quiz Scores' },
                  { id: 'retry', label: 'Retry Counts' },
                  { id: 'time', label: 'Time Spent' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-5 py-3 text-sm font-medium whitespace-nowrap ${activeTab === tab.id
                      ? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                      }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="transition-all duration-300">
                {activeTab === 'quiz' && (
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Quiz Scores
                        <span className="text-rose-500 ml-1">*</span>
                      </label>
                      <button 
                        type="button"
                        onClick={() => handleExampleClick('quiz_scores')}
                        className="text-sm text-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                      >
                        Show Example
                      </button>
                    </div>
                    <textarea
                      name="quiz_scores"
                      value={form.quiz_scores}
                      onChange={handleChange}
                      required
                      className="w-full px-5 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition min-h-[120px]"
                      placeholder="Example: Recursion: 60 (out of 100)"
                    />
                  </div>
                )}

                {activeTab === 'retry' && (
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Retry Counts
                        <span className="text-rose-500 ml-1">*</span>
                      </label>
                      <button 
                        type="button"
                        onClick={() => handleExampleClick('retry_data')}
                        className="text-sm text-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                      >
                        Show Example
                      </button>
                    </div>
                    <textarea
                      name="retry_data"
                      value={form.retry_data}
                      onChange={handleChange}
                      required
                      className="w-full px-5 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition min-h-[120px]"
                      placeholder="Example:  Recursion 3 times"
                    />
                  </div>
                )}

                {activeTab === 'time' && (
                  <div>
                    <div className="flex justify-between items-center mb-3">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Time Spent
                        <span className="text-rose-500 ml-1">*</span>
                      </label>
                      <button 
                        type="button"
                        onClick={() => handleExampleClick('time_data')}
                        className="text-sm text-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                      >
                        Show Example
                      </button>
                    </div>
                    <textarea
                      name="time_data"
                      value={form.time_data}
                      onChange={handleChange}
                      required
                      className="w-full px-5 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition min-h-[120px]"
                      placeholder="Example: 45 mins Recursion"
                    />
                  </div>
                )}
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 text-white font-bold shadow-lg hover:from-blue-700 hover:to-purple-700 transition-all disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Analyzing Your Data...
                  </>
                ) : (
                  'Evaluate My Mastery'
                )}
              </button>
            </form>
          </div>
        </div>
      ) : (
        <div className="space-y-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-800 dark:text-white">
              Your Learning Profile
            </h2>
            <p className="text-gray-600 dark:text-gray-400 text-xl">
              Learner: <span className="font-bold text-blue-600 dark:text-blue-400">{form.learner_id}</span>
            </p>
          </div>

          <ResultSummary concepts={result.results.mastery_result.concepts} />

          <div className="space-y-6">
            <h3 className="text-2xl font-bold text-gray-800 dark:text-white">Detailed Concept Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {result.results.mastery_result.concepts.map((concept, index) => (
                <ConceptCard key={index} concept={concept} />
              ))}
            </div>
          </div>

          <div className="flex flex-wrap gap-4 justify-center mt-10">
            <button
              onClick={handleStartNewEvaluation}
              className="px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-white rounded-xl hover:bg-gray-200 dark:hover:bg-gray-600 transition font-medium shadow-sm"
            >
              Start New Evaluation
            </button>
            <button
              onClick={() => setAdaptiveModalOpen(true)}
              disabled={!GapsDatas}
              className={`px-6 py-3 bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-200 rounded-xl hover:bg-blue-200 dark:hover:bg-blue-800/30 transition font-medium shadow-sm ${!GapsDatas ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              Take Adaptive Test
            </button>
            <button
              onClick={handleGetPrerequisite}
              disabled={!GapsDatas}
              className={`px-6 py-3 bg-indigo-100 dark:bg-indigo-900/20 text-indigo-800 dark:text-indigo-200 rounded-xl hover:bg-indigo-200 dark:hover:bg-indigo-800/30 transition font-medium shadow-sm ${!GapsDatas ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              View Concept Details
            </button>
            <button
              onClick={handleBooster}
              disabled={!GapsDatas || !prereqResult || !adaptiveAssessmentResult}
              className={`px-6 py-3 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200 rounded-xl hover:bg-green-200 dark:hover:bg-green-800/30 transition font-medium shadow-sm ${!GapsDatas || !prereqResult || !adaptiveAssessmentResult ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              Get Learning Boosters
            </button>
          </div>

          {!GapsDatas && (
            <div className="mt-10 text-center">
              <h3 className="text-xl font-semibold text-red-800 dark:text-red-800 mb-4">
                No Weak Concepts Found
              </h3>
            </div>
          )}
        </div>
      )}

      <div className="mt-8">
        {loading && <Loader />}
        {boosterLoading && <Loader />}
        {prereqLoading && <Loader />}
        {error && <ErrorMessage message={error} />}
      </div>

      {GapsDatas && <PrerequisiteGaps gapsData={GapsDatas} />}
      {prereqResult && <PrerequisiteResults prereqResult={prereqResult} />}
      {boosterResult && <BoosterResults boosterResult={boosterResult} />}

      <AdaptiveAssessModal
        open={adaptiveModalOpen}
        onClose={() => setAdaptiveModalOpen(false)}
        concepts={result ? result.results.mastery_result.concepts : []}
        gabsData={GapsDatas}
        onComplete={(assessmentResult) => {
          setAdaptiveAssessmentResult(assessmentResult);
          setAdaptiveModalOpen(false);
        }}
      />
    </div>
  );
}