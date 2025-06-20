export default function Loader() {
  return (
    <div className="flex justify-center items-center py-4">
      <div className="w-8 h-8 border-4 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
      <span className="ml-3 text-blue-600 dark:text-blue-300 font-semibold">Loading...</span>
    </div>
  );
} 