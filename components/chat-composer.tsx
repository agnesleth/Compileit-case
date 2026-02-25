"use client"

import type { FormEvent } from "react"
import { SendHorizonal } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export function ChatComposer({
  input,
  isLoading,
  errorMessage,
  onInputChange,
  onSubmit,
}: {
  input: string
  isLoading: boolean
  errorMessage?: string
  onInputChange: (value: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}) {
  const isDisabled = isLoading || input.trim().length === 0

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-30 px-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] sm:px-6 sm:pb-[calc(1rem+env(safe-area-inset-bottom))]">
      <div className="pointer-events-auto mx-auto w-full max-w-3xl">
        {errorMessage ? (
          <div className="mb-3 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-200">
            {errorMessage}
          </div>
        ) : null}

        <form
          onSubmit={onSubmit}
          className="flex items-center gap-2 rounded-2xl border border-slate-800 bg-slate-950/95 p-2 shadow-[0_0_36px_rgba(37,99,235,0.22)] backdrop-blur"
        >
          <Input
            value={input}
            onChange={(event) => onInputChange(event.target.value)}
            placeholder="Ställ en fråga om Compileit..."
            className="h-11 border-none bg-transparent text-base text-slate-100 placeholder:text-slate-500 focus-visible:ring-0"
            disabled={isLoading}
            autoComplete="off"
            aria-label="Meddelande till Compileits AI-assistent"
          />

          <Button
            type="submit"
            disabled={isDisabled}
            size="icon"
            className="size-11 rounded-xl bg-blue-600 text-white hover:bg-blue-500 disabled:bg-slate-700 disabled:text-slate-400"
            aria-label="Skicka fråga"
          >
            {isLoading ? (
              <span className="size-4 animate-spin rounded-full border-2 border-white/25 border-t-white" />
            ) : (
              <SendHorizonal className="size-4" />
            )}
          </Button>
        </form>

        <p className="pt-3 text-center text-xs text-slate-500">
          Drivs av Compileit • Svar baserat på compileit.com
        </p>
      </div>
    </div>
  )
}
