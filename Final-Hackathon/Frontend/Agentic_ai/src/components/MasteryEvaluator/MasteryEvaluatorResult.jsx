export default function MasteryEvaluatorResult({ result }) {
  return (
    <div className="mt-6 p-6 rounded-2xl bg-gradient-to-br from-green-100 via-blue-50 to-purple-100 dark:from-gray-800 dark:via-gray-900 dark:to-gray-800 shadow-lg border border-gray-200 dark:border-gray-700">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">Evaluation Result</h2>
      <pre className="overflow-x-auto bg-gray-900 text-green-200 rounded-lg p-4 text-sm max-h-72">
        {JSON.stringify(result, null, 2)}
      </pre>
    </div>
  );
} 