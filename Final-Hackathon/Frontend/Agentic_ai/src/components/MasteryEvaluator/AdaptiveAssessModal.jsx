import React, { useState } from 'react';

const masteryColors = {
  Weak: 'bg-rose-100 text-rose-800 border-rose-400',
  Moderate: 'bg-amber-100 text-amber-800 border-amber-400',
  Strong: 'bg-green-100 text-green-800 border-green-400',
};

const masteryIcons = {
  Weak: (
    <svg className="w-6 h-6 text-rose-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
  ),
  Moderate: (
    <svg className="w-6 h-6 text-amber-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /><path strokeLinecap="round" strokeLinejoin="round" d="M8 12h8" /></svg>
  ),
  Strong: (
    <svg className="w-6 h-6 text-green-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
  ),
};

export default function AdaptiveAssessModal({ open, onClose, concepts }) {
  const [step, setStep] = useState(1); // 1: select, 2: assess, 3: summary
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userAnswers, setUserAnswers] = useState({});
  const [grading, setGrading] = useState({});
  const [gradeResults, setGradeResults] = useState({});
  const [currentQ, setCurrentQ] = useState(0);

  if (!open) return null;

  // Step 1: Select concept
  const handleConceptClick = async (concept) => {
    setSelected(concept);
    setLoading(true);
    setError(null);
    setResult(null);
    setUserAnswers({});
    setGradeResults({});
    setCurrentQ(0);
    setStep(2);
    try {
      const response = await fetch('http://localhost:5000/api/evaluate/adaptive/questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ concept: concept.name, level: concept.mastery.toLowerCase() })
      });
      if (!response.ok) throw new Error('Failed to fetch assessment');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong');
      setStep(1);
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Answer questions
  const handleInputChange = (value) => {
    setUserAnswers((prev) => ({ ...prev, [currentQ]: value }));
  };

  const handleGrade = async () => {
    const q = result.questions[currentQ];
    setGrading((prev) => ({ ...prev, [currentQ]: true }));
    setGradeResults((prev) => ({ ...prev, [currentQ]: null }));
    try {
      const response = await fetch('http://localhost:5000/api/evaluate/adaptive/grade', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          concept: selected.name,
          question: q.question,
          ideal_answer: q.ideal_answer,
          user_answer: userAnswers[currentQ] || ''
        })
      });
      if (!response.ok) throw new Error('Failed to grade answer');
      const data = await response.json();
      setGradeResults((prev) => ({ ...prev, [currentQ]: data }));
    } catch (err) {
      setGradeResults((prev) => ({ ...prev, [currentQ]: { error: err.message || 'Grading failed' } }));
    } finally {
      setGrading((prev) => ({ ...prev, [currentQ]: false }));
    }
  };

  const handleNext = () => {
    if (currentQ < result.questions.length - 1) {
      setCurrentQ(currentQ + 1);
    } else {
      setStep(3); // summary
    }
  };

  const handlePrev = () => {
    if (currentQ > 0) setCurrentQ(currentQ - 1);
  };

  const handleRestart = () => {
    setStep(1);
    setSelected(null);
    setResult(null);
    setUserAnswers({});
    setGradeResults({});
    setCurrentQ(0);
    setError(null);
  };

  // Modal glassmorphism style
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative bg-white/80 dark:bg-gray-900/80 rounded-3xl shadow-2xl p-0 w-full max-w-4xl animate-fadeIn border border-gray-200 dark:border-gray-800 overflow-hidden flex flex-col" style={{ maxHeight: '90vh' }}>
        <button onClick={onClose} className="absolute top-5 right-6 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 text-3xl font-bold z-10">&times;</button>
        {/* Header */}
        <div className="px-8 pt-8 pb-4 border-b border-gray-100 dark:border-gray-800 bg-white/60 dark:bg-gray-900/60">
          <h2 className="text-3xl font-extrabold mb-1 text-center text-blue-700 dark:text-blue-300 tracking-tight">Adaptive Assessment</h2>
          <p className="mb-2 text-gray-600 dark:text-gray-300 text-center text-lg">Personalized, step-by-step mastery check</p>
        </div>
        
        {/* Content area with scroll */}
        <div className="flex-1 overflow-y-auto">
          {/* Step 1: Select Concept */}
          {step === 1 && (
            <div className="p-8 flex flex-col gap-6">
              <div className="text-center text-lg font-medium mb-2">Select a concept to begin:</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {concepts.filter(c => c.mastery === 'Weak' || c.mastery === 'Moderate').map((c, i) => (
                  <button
                    key={i}
                    onClick={() => handleConceptClick(c)}
                    className={`flex flex-col items-center gap-2 p-6 rounded-2xl shadow-lg border-2 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white/80 dark:bg-gray-800/80 hover:scale-105 ${masteryColors[c.mastery]}`}
                    disabled={loading}
                  >
                    <div>{masteryIcons[c.mastery]}</div>
                    <div className="text-xl font-semibold">{c.name}</div>
                    <span className={`mt-1 px-3 py-0.5 rounded-full text-xs font-bold border ${masteryColors[c.mastery]}`}>{c.mastery}</span>
                  </button>
                ))}
              </div>
              {error && <div className="text-center text-red-500 mt-2">{error}</div>}
            </div>
          )}
          
          {/* Step 2: Assessment */}
          {step === 2 && (
            <div className="p-8 flex flex-col gap-6">
              {loading && <div className="text-center text-blue-500 text-lg">Loading assessment...</div>}
              {result && result.questions && (
                <>
                  {/* Progress Bar */}
                  <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden mb-4">
                    <div className="h-full bg-blue-400 transition-all" style={{ width: `${((currentQ + 1) / result.questions.length) * 100}%` }}></div>
                  </div>
                  {/* Question Card */}
                  <div className="bg-blue-50 dark:bg-blue-900 rounded-2xl p-6 shadow flex flex-col gap-4 animate-fadeIn">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-blue-600 dark:text-blue-200 font-bold text-lg">Q{currentQ + 1} of {result.questions.length}</span>
                    </div>
                    <div className="font-semibold text-lg mb-2">{result.questions[currentQ].question}</div>
                    <textarea
                      className="w-full p-3 rounded-xl border border-blue-300 text-gray-900 text-base focus:ring-2 focus:ring-blue-400 min-h-[100px]"
                      placeholder="Type your answer..."
                      value={userAnswers[currentQ] || ''}
                      onChange={e => handleInputChange(e.target.value)}
                      disabled={grading[currentQ]}
                    />
                    <div className="flex gap-3 mt-2 flex-wrap">
                      <button
                        className="px-5 py-2 rounded-xl bg-blue-600 text-white font-semibold hover:bg-blue-700 transition disabled:opacity-50 flex items-center gap-2"
                        onClick={handleGrade}
                        disabled={grading[currentQ] || !userAnswers[currentQ] || gradeResults[currentQ]}
                      >
                        {grading[currentQ] ? (
                          <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" /></svg>
                        ) : (
                          <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                        )}
                        Submit
                      </button>
                      <button
                        className="px-5 py-2 rounded-xl bg-gray-200 text-gray-700 font-semibold hover:bg-gray-300 transition disabled:opacity-50"
                        onClick={handlePrev}
                        disabled={currentQ === 0}
                      >Back</button>
                      <button
                        className="px-5 py-2 rounded-xl bg-blue-100 text-blue-700 font-semibold hover:bg-blue-200 transition disabled:opacity-50"
                        onClick={handleNext}
                        disabled={!gradeResults[currentQ]}
                      >{currentQ === result.questions.length - 1 ? 'Finish' : 'Next'}</button>
                    </div>
                    {/* Feedback */}
                    {gradeResults[currentQ] && (
                      <div className="mt-4 text-base animate-fadeIn">
                        {gradeResults[currentQ].error ? (
                          <div className="flex items-center gap-2 text-red-500 font-semibold">
                            <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                            {gradeResults[currentQ].error}
                          </div>
                        ) : (
                          <div className="space-y-2">
                            <div className="flex gap-2 flex-wrap">
                              <span className={`px-3 py-1 rounded-full text-xs font-bold ${gradeResults[currentQ].score?.correctness === 1 ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-700'}`}>Correctness: {gradeResults[currentQ].score?.correctness ?? '-'}</span>
                              <span className={`px-3 py-1 rounded-full text-xs font-bold ${gradeResults[currentQ].score?.completeness === 1 ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-700'}`}>Completeness: {gradeResults[currentQ].score?.completeness ?? '-'}</span>
                              <span className={`px-3 py-1 rounded-full text-xs font-bold ${gradeResults[currentQ].score?.reasoning === 1 ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-700'}`}>Reasoning: {gradeResults[currentQ].score?.reasoning ?? '-'}</span>
                            </div>
                            <div className="bg-blue-100 text-blue-900 rounded-xl p-3 border border-blue-200 flex items-start gap-2">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mt-0.5 text-blue-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M12 20a8 8 0 100-16 8 8 0 000 16z" /></svg>
                              <span>{gradeResults[currentQ].feedback}</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
          
          {/* Step 3: Summary */}
          {step === 3 && (
            <div className="p-8 flex flex-col gap-6 items-center animate-fadeIn">
              <div className="text-2xl font-bold text-blue-700 dark:text-blue-200 mb-2">Assessment Complete!</div>
              <div className="w-full bg-white/70 dark:bg-gray-800/70 rounded-2xl shadow p-6 flex flex-col gap-4">
                {result.questions.map((q, idx) => (
                  <div key={idx} className="flex flex-col gap-1 pb-4 border-b border-gray-200 dark:border-gray-700 last:border-0">
                    <div className="font-semibold text-blue-900 dark:text-blue-100">Q{idx + 1}: {q.question}</div>
                    <div className="text-gray-700 dark:text-gray-200">Your answer: <span className="font-medium">{userAnswers[idx]}</span></div>
                    {gradeResults[idx] && !gradeResults[idx].error && (
                      <div className="flex gap-2 mt-1 flex-wrap">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${gradeResults[idx].score?.correctness === 1 ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-700'}`}>C: {gradeResults[idx].score?.correctness ?? '-'}</span>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${gradeResults[idx].score?.completeness === 1 ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-700'}`}>Cp: {gradeResults[idx].score?.completeness ?? '-'}</span>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${gradeResults[idx].score?.reasoning === 1 ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-700'}`}>R: {gradeResults[idx].score?.reasoning ?? '-'}</span>
                      </div>
                    )}
                    {gradeResults[idx] && gradeResults[idx].feedback && (
                      <div className="text-blue-700 dark:text-blue-200 text-sm mt-1">{gradeResults[idx].feedback}</div>
                    )}
                    {gradeResults[idx] && gradeResults[idx].error && (
                      <div className="text-red-500 text-sm mt-1">{gradeResults[idx].error}</div>
                    )}
                  </div>
                ))}
              </div>
              <button
                className="mt-6 px-8 py-3 rounded-2xl bg-blue-600 text-white font-bold text-lg hover:bg-blue-700 transition shadow-lg"
                onClick={handleRestart}
              >Try Another Concept</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}