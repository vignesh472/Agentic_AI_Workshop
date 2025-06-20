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
  quiz_scores: "Loops: 85, Recursion: 60, Data Structures: 40",
  coding_logs: "3 attempts on Loops, 7 on Recursion, 10 on Data Structures",
  retry_data: "Loops retried once, Recursion 3 times, Data Structures 5 times",
  time_data: "20.5 mins on Loops, 45 mins Recursion, 50 mins Data Structures"
};

const MasteryLevel = ({ mastery }) => {
  const levels = {
    'Strong': { color: 'bg-emerald-100 text-emerald-800', icon: '✓' },
    'Moderate': { color: 'bg-amber-100 text-amber-800', icon: '~' },
    'Weak': { color: 'bg-rose-100 text-rose-800', icon: '!' }
  };

  const level = levels[mastery] || levels['Weak'];

  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${level.color}`}>
      <span className="mr-1">{level.icon}</span>
      {mastery}
    </span>
  );
};

const ConceptCard = ({ concept }) => {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 border-l-4 border-blue-500">
      <div className="flex justify-between items-start">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white">{concept.name}</h3>
        <MasteryLevel mastery={concept.mastery} />
      </div>
      <p className="mt-2 text-gray-600 dark:text-gray-300">{concept.reason}</p>
      <div className="mt-4 pt-4 border-t border-gray-100 dark:border-gray-700">
        <div className="flex items-center text-sm text-gray-500 dark:text-gray-400">
          <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Recommendation: {getRecommendation(concept.mastery, concept.name)}</span>
        </div>
      </div>
    </div>
  );
};

const getRecommendation = (mastery, topic) => {
  const recommendations = {
    'Strong': `Continue applying ${topic} concepts in more complex problems to reinforce your understanding.`,
    'Moderate': `Review ${topic} fundamentals and practice with additional exercises to strengthen your skills.`,
    'Weak': `Start with basic ${topic} tutorials and complete foundational exercises before moving to advanced topics.`
  };
  return recommendations[mastery] || `Focus on learning ${topic} concepts through structured practice.`;
};

const ResultSummary = ({ concepts }) => {
  const masteryCounts = concepts.reduce((acc, concept) => {
    acc[concept.mastery] = (acc[concept.mastery] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-gray-700 dark:to-gray-800 rounded-lg p-6 mb-6">
      <h2 className="text-xl font-bold text-gray-800 dark:text-white mb-4">Mastery Summary</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-emerald-100 dark:border-gray-700">
          <div className="text-emerald-600 dark:text-emerald-400 font-bold text-2xl">{masteryCounts['Strong'] || 0}</div>
          <div className="text-gray-500 dark:text-gray-400 text-sm">Strong Concepts</div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-amber-100 dark:border-gray-700">
          <div className="text-amber-600 dark:text-amber-400 font-bold text-2xl">{masteryCounts['Moderate'] || 0}</div>
          <div className="text-gray-500 dark:text-gray-400 text-sm">Moderate Concepts</div>
        </div>
        <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-sm border border-rose-100 dark:border-gray-700">
          <div className="text-rose-600 dark:text-rose-400 font-bold text-2xl">{masteryCounts['Weak'] || 0}</div>
          <div className="text-gray-500 dark:text-gray-400 text-sm">Weak Concepts</div>
        </div>
      </div>
    </div>
  );
};

const PrerequisiteGaps = ({ gapsData }) => {
  if (!gapsData) return null;

  // If there's a message (no gaps), show a positive message
  if (gapsData.message) {
    return (
      <div className="mt-8 bg-emerald-50 dark:bg-emerald-900 rounded-xl p-6 flex items-center justify-center shadow">
        <svg className="w-8 h-8 text-emerald-500 mr-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
        <span className="text-lg font-semibold text-emerald-700 dark:text-emerald-200">{gapsData.message}</span>
      </div>
    );
  }

  // Otherwise, show the gaps
  return (
    <div className="mt-8 bg-yellow-50 dark:bg-yellow-900 rounded-xl p-6 shadow-lg">
      <h2 className="text-2xl font-bold text-yellow-800 dark:text-yellow-200 mb-2 flex items-center">
        <svg className="w-7 h-7 mr-2 text-yellow-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Prerequisite Gaps Detected
      </h2>
      <p className="text-yellow-700 dark:text-yellow-300 mb-4">
        These are foundational topics you should review to strengthen your understanding:
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-4">
        {gapsData.gaps.map((gap, idx) => (
          <div key={idx} className="flex items-center bg-white dark:bg-gray-800 border-l-4 border-yellow-400 rounded-lg p-4 shadow-sm">
            <svg className="w-6 h-6 text-yellow-400 mr-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="font-medium text-gray-800 dark:text-gray-100">{gap}</span>
            {/* Optional: <button className="ml-auto text-xs text-blue-500 hover:underline">Learn More</button> */}
          </div>
        ))}
      </div>
      <div className="text-center mt-4">
        <span className="inline-block bg-yellow-200 dark:bg-yellow-700 text-yellow-900 dark:text-yellow-100 px-4 py-2 rounded-full font-semibold">
          Start with these topics to build a stronger foundation!
        </span>
      </div>
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
console.log("boosterResult",boosterResult)
console.log("prereqResult",prereqResult)
  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const parseInputToDict = (input) => {
    const dict = {};
    input.split(',').forEach(entry => {
      const [key, value] = entry.split(':').map(s => s.trim());
      if (key && !isNaN(parseFloat(value))) {
        dict[key] = parseFloat(value);
      }
    });
    return dict;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    
    try {
      const res = await evaluateMastery({
        learner_id: form.learner_id,
        quiz_scores: parseInputToDict(form.quiz_scores),
        coding_logs: parseInputToDict(form.coding_logs),
        retry_data: parseInputToDict(form.retry_data),
        time_data: parseInputToDict(form.time_data),
      });
      
      setResult(res);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

 const handleFindGap = async () => {
  if (!result) return;
  
  setLoading(true);
  setError(null);
  
  try {
    // Get weak topics as an array of strings
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
      body: JSON.stringify(weakTopics), // Send just the array of topics
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
      const concepts = result.evaluation.concepts.map(c => c.name);
      const response = await fetch('http://localhost:5000/api/evaluate/booster-recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ concepts, preference: 'interactive' }),
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
      const weakTopics = result.evaluation.concepts
        .filter(concept => concept.mastery === 'Weak')
        .map(concept => concept.name);
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
        body: JSON.stringify({ concepts: weakTopics }),
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

  return (
    <div className="min-w-8xl mx-auto p-6 bg-white dark:bg-gray-900 rounded-xl shadow-lg">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 dark:text-white mb-2">Natural Language Mastery Evaluator</h1>
        <p className="text-gray-600 dark:text-gray-300">
          Describe your performance in natural language - we'll handle the formatting
        </p>
      </div>

      {!result ? (
        <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg mb-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Learner ID
              </label>
              <input
                name="learner_id"
                value={form.learner_id}
                onChange={handleChange}
                required
                className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
                placeholder="Enter your learner ID (e.g., learner_123)"
              />
            </div>

            <div className="flex border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
              {[
                { id: 'quiz', label: 'Quiz Scores' },
                { id: 'coding', label: 'Coding Attempts' },
                { id: 'retry', label: 'Retry Counts' },
                { id: 'time', label: 'Time Spent' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-4 py-2 text-sm font-medium whitespace-nowrap ${activeTab === tab.id
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
                  <div className="flex justify-between items-center mb-1">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Quiz Scores
                    </label>
                    <button 
                      type="button"
                      onClick={() => handleExampleClick('quiz_scores')}
                      className="text-xs text-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                    >
                      Insert Example
                    </button>
                  </div>
                  <textarea
                    name="quiz_scores"
                    value={form.quiz_scores}
                    onChange={handleChange}
                    required
                    className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition min-h-[100px]"
                    placeholder="Example: Loops: 85, Recursion: 60, Data Structures: 40"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Describe your quiz scores by topic (e.g., "I got 85 on Loops, 60 on Recursion")
                  </p>
                </div>
              )}

              {activeTab === 'coding' && (
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Coding Attempts
                    </label>
                    <button 
                      type="button"
                      onClick={() => handleExampleClick('coding_logs')}
                      className="text-xs text-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                    >
                      Insert Example
                    </button>
                  </div>
                  <textarea
                    name="coding_logs"
                    value={form.coding_logs}
                    onChange={handleChange}
                    required
                    className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition min-h-[100px]"
                    placeholder="Example: 3 attempts on Loops, 7 on Recursion, 10 on Data Structures"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Describe how many times you attempted coding exercises by topic
                  </p>
                </div>
              )}

              {activeTab === 'retry' && (
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Retry Counts
                    </label>
                    <button 
                      type="button"
                      onClick={() => handleExampleClick('retry_data')}
                      className="text-xs text-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                    >
                      Insert Example
                    </button>
                  </div>
                  <textarea
                    name="retry_data"
                    value={form.retry_data}
                    onChange={handleChange}
                    required
                    className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition min-h-[100px]"
                    placeholder="Example: Loops retried once, Recursion 3 times, Data Structures 5 times"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Describe how many times you retried exercises by topic
                  </p>
                </div>
              )}

              {activeTab === 'time' && (
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                      Time Spent
                    </label>
                    <button 
                      type="button"
                      onClick={() => handleExampleClick('time_data')}
                      className="text-xs text-blue-500 hover:text-blue-700 dark:hover:text-blue-400"
                    >
                      Insert Example
                    </button>
                  </div>
                  <textarea
                    name="time_data"
                    value={form.time_data}
                    onChange={handleChange}
                    required
                    className="w-full p-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition min-h-[100px]"
                    placeholder="Example: 20.5 minutes on Loops, 45 mins Recursion, 50 mins Data Structures"
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Describe how much time you spent on each topic (in minutes)
                  </p>
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 px-6 rounded-lg bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium shadow-md hover:from-blue-700 hover:to-indigo-700 transition-all disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center"
            >
              {loading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </>
              ) : (
                'Evaluate My Mastery'
              )}
            </button>
          </form>
        </div>
      ) : (
        <div className="space-y-8">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
              Mastery Evaluation Results
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              For learner: <span className="font-medium text-blue-600 dark:text-blue-400">{form.learner_id}</span>
            </p>
          </div>

          <ResultSummary concepts={result.evaluation.concepts} />

          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Detailed Concept Analysis</h3>
            <div className="grid grid-cols-1 gap-4">
              {result.evaluation.concepts.map((concept, index) => (
                <ConceptCard key={index} concept={concept} />
              ))}
            </div>
          </div>

          <div className="flex justify-center">
            <button
              onClick={() => setResult(null)}
              className="px-6 py-2 bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-white rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition"
            >
              Perform Another Evaluation
            </button>
            <button
              onClick={() => handleFindGap()}
              className="px-6 py-2 ml-2 bg-orange-100 dark:bg-gray-700 text-gray-800 dark:text-white rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition"
            >
              Find Gaps
            </button>
            <button
              onClick={() => setAdaptiveModalOpen(true)}
              className="px-6 py-2 bg-blue-100 dark:bg-blue-700 text-blue-800 dark:text-white rounded-lg hover:bg-blue-200 dark:hover:bg-blue-600 transition ml-2"
            >
              Find the Level
            </button>
            <button
              onClick={handleBooster}
              className="px-6 py-2 bg-green-100 dark:bg-green-700 text-green-800 dark:text-white rounded-lg hover:bg-green-200 dark:hover:bg-green-600 transition ml-2"
            >
              Booster
            </button>
            <button
              onClick={handleGetPrerequisite}
              className="px-6 py-2 bg-yellow-100 dark:bg-yellow-700 text-yellow-800 dark:text-white rounded-lg hover:bg-yellow-200 dark:hover:bg-yellow-600 transition ml-2"
            >
              Get Prerequisite
            </button>
          </div>
        </div>
      )}

      <div className="mt-8 transition-all duration-300">
        {loading && <Loader />}
        {boosterLoading && <Loader />}
        {prereqLoading && <Loader />}
        {error && <ErrorMessage message={error} />}
      </div>

      {GapsData && <PrerequisiteGaps gapsData={GapsData} />}
      {boosterResult && (
        <div className="mt-8 bg-green-50 dark:bg-green-900 rounded-xl p-6 shadow-lg">
          <div className="flex items-center mb-4">
            <svg className="w-7 h-7 mr-2 text-green-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2 className="text-2xl font-bold text-green-800 dark:text-green-200">Booster Recommendations</h2>
            {boosterResult.status === 'success' && (
              <span className="ml-4 inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-200 text-emerald-800 dark:bg-emerald-700 dark:text-emerald-100">Success</span>
            )}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {boosterResult.boosters && boosterResult.boosters.map((booster, idx) => (
              <div key={idx} className="bg-white dark:bg-gray-800 rounded-lg shadow p-5 border-l-4 border-green-400 flex flex-col h-full">
                <div className="flex items-center mb-2">
                  <span className="inline-block bg-green-100 dark:bg-green-700 text-green-800 dark:text-green-100 px-3 py-1 rounded-full text-xs font-semibold mr-2">
                    {booster.booster_type.replace(/(^|\s)\S/g, l => l.toUpperCase())}
                  </span>
                  <span className="inline-block bg-blue-100 dark:bg-blue-700 text-blue-800 dark:text-blue-100 px-3 py-1 rounded-full text-xs font-semibold">
                    {booster.format}
                  </span>
                </div>
                <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-1 flex items-center">
                  <svg className="w-5 h-5 mr-1 text-blue-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l9-5-9-5-9 5 9 5z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l6.16-3.422A12.083 12.083 0 0112 21.5a12.083 12.083 0 01-6.16-10.922L12 14z" />
                  </svg>
                  {booster.concept}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 mb-3 flex-1">{booster.description}</p>
                <div className="flex items-center mt-auto">
                  <svg className="w-4 h-4 mr-1 text-amber-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 17l4 4 4-4m-4-5v9" />
                  </svg>
                  <span className="text-sm text-amber-700 dark:text-amber-300 font-medium">{booster.estimated_duration}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
  {prereqResult && (
  <div className="mt-8 bg-yellow-50 dark:bg-yellow-900 rounded-xl p-6 shadow-lg">
    <div className="flex items-center mb-4">
      <svg className="w-7 h-7 mr-2 text-yellow-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <h2 className="text-2xl font-bold text-yellow-800 dark:text-yellow-200">Prerequisite Recommendations</h2>
    </div>

    {prereqResult.message ? (
      <div className="text-yellow-700 dark:text-yellow-200 font-medium">{prereqResult.message}</div>
    ) : (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {prereqResult.retrieved && prereqResult.retrieved.map((prereq, idx) => (
          <div 
            key={idx} 
            className={`
              bg-white dark:bg-gray-800 rounded-lg shadow p-5 border-l-4 border-yellow-400
              ${prereqResult.retrieved.length === 1 ? 'col-span-full' : ''}
              ${prereqResult.retrieved.length === 2 ? 'sm:col-span-1' : ''}
            `}
          >
            <div className="flex items-center mb-2">
              <span className="inline-block bg-yellow-100 dark:bg-yellow-700 text-yellow-800 dark:text-yellow-100 px-3 py-1 rounded-full text-xs font-semibold mr-2">
                {prereq.concept}
              </span>
              <span className="inline-block bg-amber-200 dark:bg-amber-700 text-amber-800 dark:text-amber-100 px-2 py-0.5 rounded-full text-xs font-semibold">
                {prereq.source_count} Source{prereq.source_count !== 1 ? 's' : ''}
              </span>
            </div>
            <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-1 flex items-center">
              <svg className="w-5 h-5 mr-1 text-yellow-500" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l9-5-9-5-9 5 9 5z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 14l6.16-3.422A12.083 12.083 0 0112 21.5a12.083 12.083 0 01-6.16-10.922L12 14z" />
              </svg>
              {prereq.concept}
            </h3>
            <p className="text-gray-700 dark:text-gray-200 mb-3">{prereq.summary}</p>
            <div className="flex items-start mt-2 bg-yellow-100 dark:bg-yellow-800 rounded p-3">
              <svg className="w-5 h-5 mr-2 text-yellow-600 dark:text-yellow-300 mt-0.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <span className="block text-xs font-semibold text-yellow-800 dark:text-yellow-200 mb-1">Real-World Example</span>
                <span className="text-xs text-yellow-900 dark:text-yellow-100">{prereq.example}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    )}
  </div>
)}

      <AdaptiveAssessModal
        open={adaptiveModalOpen}
        onClose={() => setAdaptiveModalOpen(false)}
        concepts={result ? result.evaluation.concepts : []}
      />
    </div>
  );
}