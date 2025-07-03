import { useState } from 'react'
import './App.css'
import MasteryEvaluator from './components/MasteryEvaluator/MasteryEvaluatorForm'

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="min-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <header className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-24 h-24 rounded-3xl bg-gradient-to-tr from-purple-600 to-blue-500 shadow-lg mb-6">
            <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-5.342m0 0a50.655 50.655 0 013.843-1.586M12 13.49a50.657 50.657 0 00-7.74-5.342M12 3.49a50.66 50.66 0 013.843 1.586m-7.74 5.342a50.639 50.639 0 00-3.843 1.586" />
            </svg>
          </div>
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-purple-600 to-blue-500">
              Mastery Metrics
            </span>
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Transform your learning journey with AI-powered insights and personalized recommendations
          </p>
        </header>
        
        <div className="bg-white dark:bg-gray-800 rounded-3xl shadow-2xl overflow-hidden border border-gray-200 dark:border-gray-700">
          <MasteryEvaluator />
        </div>
        
        {/* <footer className="mt-16 text-center text-gray-500 dark:text-gray-400">
          <p className="text-sm">© {new Date().getFullYear()} Mastery Metrics. All rights reserved.</p>
        </footer> */}
      </div>
    </div>
  )
}