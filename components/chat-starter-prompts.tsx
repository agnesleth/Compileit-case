"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export function ChatStarterPrompts({
  prompts,
  disabled,
  onSelectPrompt,
}: {
  prompts: string[]
  disabled: boolean
  onSelectPrompt: (prompt: string) => void
}) {
  return (
    <div className="mt-8 grid w-full max-w-2xl grid-cols-1 gap-3 text-left sm:grid-cols-2">
      {prompts.map((prompt, index) => (
        <Button
          key={prompt}
          type="button"
          variant="outline"
          disabled={disabled}
          onClick={() => onSelectPrompt(prompt)}
          className={cn(
            "h-auto min-h-16 whitespace-normal rounded-2xl border-slate-800 bg-slate-900/80 px-5 py-4 text-sm leading-6 text-slate-200 hover:bg-slate-900 hover:text-white",
            index === prompts.length - 1 && prompts.length % 2 !== 0
              ? "sm:col-span-2 sm:mx-auto sm:w-[calc(50%-0.375rem)]"
              : ""
          )}
        >
          {prompt}
        </Button>
      ))}
    </div>
  )
}
