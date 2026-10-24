import { useDropzone } from 'react-dropzone'

const MAX_MB = 10

interface Props {
  previewUrl: string | null
  fileName: string | null
  disabled: boolean
  onFile: (file: File) => void
  onReject: (message: string) => void
}

export default function UploadZone({ previewUrl, fileName, disabled, onFile, onReject }: Props) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'] },
    maxFiles: 1,
    maxSize: MAX_MB * 1024 * 1024,
    disabled,
    onDropAccepted: (files) => onFile(files[0]),
    onDropRejected: (rejections) => {
      const code = rejections[0]?.errors[0]?.code
      onReject(code === 'file-too-large' ? `File is larger than ${MAX_MB} MB.` : 'Please upload a single JPG, PNG, BMP or TIFF image.')
    },
  })

  return (
    <div
      {...getRootProps()}
      className={`relative flex min-h-72 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed transition-colors ${
        isDragActive ? 'border-teal-500 bg-teal-50' : 'border-slate-300 bg-white hover:border-teal-400 hover:bg-slate-50'
      } ${disabled ? 'pointer-events-none opacity-60' : ''}`}
    >
      <input {...getInputProps()} aria-label="Upload image" />
      {previewUrl ? (
        <>
          <img src={previewUrl} alt="Uploaded image" className="max-h-80 w-full object-contain p-3" />
          <div className="w-full border-t border-slate-200 bg-slate-50 px-4 py-2 text-center text-sm text-slate-600">
            <span className="font-medium text-slate-800">{fileName}</span> · click or drop to replace
          </div>
        </>
      ) : (
        <div className="px-6 py-10 text-center">
          <svg className="mx-auto mb-4 h-12 w-12 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-base font-medium text-slate-800">
            {isDragActive ? 'Drop the image here' : 'Drag & drop an image'}
          </p>
          <p className="mt-1 text-sm text-slate-500">or click to browse · JPG, PNG, BMP, TIFF · max {MAX_MB} MB</p>
        </div>
      )}
    </div>
  )
}
