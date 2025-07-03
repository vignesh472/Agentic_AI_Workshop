import React, { useState } from 'react';

const masteryColors = {
  Weak: 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20',
  Moderate: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20',
  Strong: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
};

const masteryIcons = {
  Weak: (
    <svg className="w-5 h-5 text-rose-500" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  Moderate: (
    <svg className="w-5 h-5 text-amber-500" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
    </svg>
  ),
  Strong: (
    <svg className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
};

export default function AdaptiveAssessModal({ open, onClose, concepts, gabsData, onComplete }) {
  const [step, setStep] = useState(1);
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userAnswers, setUserAnswers] = useState({});
  const [grading, setGrading] = useState({});
  const [gradeResults, setGradeResults] = useState({});
  const [currentQ, setCurrentQ] = useState(0);
  const [selectedOption, setSelectedOption] = useState(null);

  if (!open) return null;

  const handleConceptClick = async (concept) => {
    setSelected(concept);
    setLoading(true);
    setError(null);
    setResult(null);
    setUserAnswers({});
    setGradeResults({});
    setCurrentQ(0);
    setSelectedOption(null);
    setStep(2);

    let matchingCategoryConcepts = [];

    if (gabsData?.[concept.name]) {
      matchingCategoryConcepts = gabsData[concept.name];
    } else {
      const foundEntry = Object.entries(gabsData || {}).find(
        ([_, concepts]) =>
          concepts.some(
            c => c.trim().toLowerCase() === concept.name.trim().toLowerCase()
          )
      );

      if (foundEntry) {
        matchingCategoryConcepts = foundEntry[1];
      }
    }

    const conceptPayload = {
      concept: concept.name,
      level: concept.mastery?.toLowerCase() || 'beginner',
      category: matchingCategoryConcepts.join(', ')
    };

    try {
      const response = await fetch('http://localhost:5000/api/evaluate/adaptive/questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(conceptPayload)
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

  const handleOptionSelect = (optionKey) => {
    setSelectedOption(optionKey);
    setUserAnswers((prev) => ({ ...prev, [currentQ]: optionKey }));
  };

  // const handleGrade = async () => {
  //   const q = result.questions[currentQ];
  //   setGrading((prev) => ({ ...prev, [currentQ]: true }));
  //   setGradeResults((prev) => ({ ...prev, [currentQ]: null }));
    
  //   try {
  //     const response = await fetch('http://localhost:5000/api/evaluate/adaptive/grade', {
  //       method: 'POST',
  //       headers: { 'Content-Type': 'application/json' },
  //       body: JSON.stringify({
  //         concept: selected.name,
  //         question: q.question,
  //         ideal_answer: q.correct_option,
  //         user_answer: userAnswers[currentQ] || ''
  //       })
  //     });
      
  //     if (!response.ok) throw new Error('Failed to grade answer');
  //     const data = await response.json();
  //     setGradeResults((prev) => ({ ...prev, [currentQ]: data }));
  //   } catch (err) {
  //     setGradeResults((prev) => ({ ...prev, [currentQ]: { error: err.message || 'Grading failed' } }));
  //   } finally {
  //     setGrading((prev) => ({ ...prev, [currentQ]: false }));
  //   }
  // };


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
        concept_data: selected, // Send the entire selected concept object
        question: q.question,
        question_data: q, // Send the entire question object
        ideal_answer: q.correct_option,
        user_answer: userAnswers[currentQ] || '',
        user_selected_option: {  // Structured data about the selected option
          key: userAnswers[currentQ],
          value: q.options[userAnswers[currentQ]]
        }
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
      setSelectedOption(userAnswers[currentQ + 1] || null);
    } else {
      setStep(3);
    }
  };

  const handlePrev = () => {
    if (currentQ > 0) {
      setCurrentQ(currentQ - 1);
      setSelectedOption(userAnswers[currentQ - 1] || null);
    }
  };

  const handleRestart = () => {
    if (onComplete && typeof onComplete === 'function') {
      onComplete({
        concept: selected.name,
        answers: userAnswers,
        scores: gradeResults,
      });
    }
    setStep(1);
    setSelected(null);
    setResult(null);
    setUserAnswers({});
    setGradeResults({});
    setCurrentQ(0);
    setSelectedOption(null);
    setError(null);
  };

  const getOptionLetter = (index) => {
    return String.fromCharCode(65 + index);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="relative bg-white dark:bg-slate-800 rounded-2xl shadow-xl w-full max-w-2xl border border-slate-200 dark:border-slate-700 overflow-hidden flex flex-col max-h-[90vh]">
        <button onClick={onClose} className="absolute top-4 right-4 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 text-2xl font-bold z-10">
          &times;
        </button>
        
        <div className="p-6 border-b border-slate-200 dark:border-slate-700">
          <h2 className="text-xl font-bold text-center text-slate-800 dark:text-white">Adaptive Assessment</h2>
          <p className="text-center text-slate-500 dark:text-slate-400 text-sm mt-1">
            {step === 1 ? 'Select a concept to assess' : 
             step === 2 ? `Assessing: ${selected?.name}` : 
             'Assessment complete'}
          </p>
        </div>
        
        <div className="flex-1 overflow-y-auto p-6">
          {step === 1 && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {concepts.filter(c => c.mastery === 'Weak' || c.mastery === 'Moderate').map((c, i) => (
                  <button
                    key={i}
                    onClick={() => handleConceptClick(c)}
                    disabled={loading}
                    className={`p-4 rounded-xl border ${masteryColors[c.mastery]} flex flex-col items-center transition hover:shadow-md`}
                  >
                    <div className="mb-2">
                      {masteryIcons[c.mastery]}
                    </div>
                    <div className="font-medium text-slate-800 dark:text-white">{c.name}</div>
                    <div className="text-xs mt-1 text-slate-500 dark:text-slate-400">{c.mastery} Mastery</div>
                  </button>
                ))}
              </div>
              {error && <div className="text-red-500 text-center">{error}</div>}
            </div>
          )}
          
          {step === 2 && (
            <div className="space-y-6">
              {loading && <div className="text-center py-8">Loading assessment...</div>}
              {result && result.questions && (
                <>
                  <div className="w-full bg-slate-100 dark:bg-slate-700 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${((currentQ + 1) / result.questions.length) * 100}%` }}
                    ></div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-slate-500 dark:text-slate-400">
                        Question {currentQ + 1} of {result.questions.length}
                      </span>
                      <span className="text-xs px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-slate-600 dark:text-slate-300">
                        Level {result.questions[currentQ].level}
                      </span>
                    </div>
                    
                    <div className="text-lg font-medium text-slate-800 dark:text-white">
                      {result.questions[currentQ].question}
                    </div>
                    
                    <div className="space-y-3 mt-4">
                      {Object.entries(result.questions[currentQ].options).map(([key, value], index) => (
                        <button
                          key={key}
                          onClick={() => handleOptionSelect(key)}
                          className={`w-full text-left p-4 rounded-lg border transition-all ${
                            selectedOption === key 
                              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
                              : 'border-slate-300 dark:border-slate-600 hover:border-blue-300 dark:hover:border-blue-700'
                          }`}
                        >
                          <div className="flex items-center">
                            <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-3 ${
                              selectedOption === key 
                                ? 'bg-blue-500 text-white' 
                                : 'bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                            }`}>
                              {getOptionLetter(index)}
                            </div>
                            <div className="text-slate-800 dark:text-white">{value}</div>
                          </div>
                        </button>
                      ))}
                    </div>
                    
                    <div className="flex gap-3 pt-2">
                      <button
                        onClick={handleGrade}
                        disabled={grading[currentQ] || !selectedOption || gradeResults[currentQ]}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition flex items-center gap-2 disabled:opacity-50"
                      >
                        {grading[currentQ] ? (
                          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                        ) : (
                          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                        Submit
                      </button>
                      
                      <button
                        onClick={handlePrev}
                        disabled={currentQ === 0}
                        className="px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-white rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition disabled:opacity-50"
                      >
                        Back
                      </button>
                      
                      <button
                        onClick={handleNext}
                        disabled={!gradeResults[currentQ]}
                        className="px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-800 dark:text-white rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition disabled:opacity-50 ml-auto"
                      >
                        {currentQ === result.questions.length - 1 ? 'Finish' : 'Next'}
                      </button>
                    </div>
                    
                    {gradeResults[currentQ] && (
                      <div className={`mt-4 p-4 rounded-lg border ${
                        gradeResults[currentQ].is_correct 
                          ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-100 dark:border-emerald-900/50' 
                          : 'bg-rose-50 dark:bg-rose-900/20 border-rose-100 dark:border-rose-900/50'
                      }`}>
                        {gradeResults[currentQ].error ? (
                          <div className="text-red-500">{gradeResults[currentQ].error}</div>
                        ) : (
                          <div className={gradeResults[currentQ].is_correct ? 'text-emerald-800 dark:text-emerald-200' : 'text-rose-800 dark:text-rose-200'}>
                            <div className="font-medium mb-2">
                              {gradeResults[currentQ].is_correct ? 'Correct!' : 'Incorrect'}
                            </div>
                            <p className="mb-2">{gradeResults[currentQ].feedback}</p>
                            <div className="text-sm font-medium">Explanation:</div>
                            <p className="text-sm">{result.questions[currentQ].explanation}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
          
          {step === 3 && (
            <div className="space-y-6">
              <div className="text-center py-4">
                <svg className="w-12 h-12 text-emerald-500 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3 className="text-xl font-bold text-slate-800 dark:text-white">Assessment Complete</h3>
                <p className="text-slate-500 dark:text-slate-400 mt-1">You've completed the assessment for {selected?.name}</p>
              </div>
              
              <div className="space-y-4">
                {result.questions.map((q, idx) => (
                  <div key={idx} className="border-b border-slate-200 dark:border-slate-700 pb-4 last:border-0">
                    <div className="font-medium text-slate-800 dark:text-white">Q{idx + 1}: {q.question}</div>
                    <div className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                      Your answer: {userAnswers[idx] ? `${userAnswers[idx]}. ${q.options[userAnswers[idx]]}` : 'No answer'}
                    </div>
                    <div className="text-sm text-slate-600 dark:text-slate-300 mt-1">
                      Correct answer: {q.correct_option}. {q.options[q.correct_option]}
                    </div>
                    {gradeResults[idx] && !gradeResults[idx].error && (
                      <div className={`mt-2 text-sm ${
                        gradeResults[idx].is_correct ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                      }`}>
                        <span className="font-medium">Feedback:</span> {gradeResults[idx].feedback}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              
              <div className="mt-6">
                <button
                  onClick={handleRestart}
                  className="w-full py-3 px-6 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
                >
                  Start New Assessment
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}