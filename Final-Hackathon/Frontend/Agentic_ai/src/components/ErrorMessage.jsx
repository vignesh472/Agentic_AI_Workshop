export default function ErrorMessage({ message }) {
  return (
    <div className="flex items-center bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mt-4" role="alert">
      <svg className="w-5 h-5 mr-2 text-red-500" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
      <span className="font-semibold">{message}</span>
    </div>
  );
} 