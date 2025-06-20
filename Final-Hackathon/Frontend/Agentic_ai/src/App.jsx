import { useState } from 'react'
import './App.css'
import MasteryEvaluator from './components/MasteryEvaluator/MasteryEvaluatorForm'

export default function App() {
  return (
    <div className="min-h-screen min-w-[1000px] l bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-900 dark:to-gray-800">
      <div className="min-w-[1000px] mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <header className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-white dark:bg-gray-800 shadow-lg mb-6 border-4 border-blue-100 dark:border-gray-700">
            <svg className="w-10 h-10 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
            Mastery Evaluator
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            AI-powered personalized learning assessment that adapts to your unique learning patterns and provides actionable insights.
          </p>
        </header>
        
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
          <MasteryEvaluator />
        </div>
        
        <footer className="mt-16 text-center text-gray-500 dark:text-gray-400 text-sm">
          <p>Empowering learners through intelligent assessment • {new Date().getFullYear()}</p>
        </footer>
      </div>
    </div>
  )
}