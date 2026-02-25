"use client"

import { useChat } from "@ai-sdk/react"
import { DefaultChatTransport } from "ai"
import type { FormEvent } from "react"
import { useMemo, useState } from "react"

import { ChatComposer } from "@/components/chat-composer"
import { ChatMessageList } from "@/components/chat-message-list"
import { ChatStarterPrompts } from "@/components/chat-starter-prompts"

const STARTER_PROMPTS = [
  "Vad erbjuder ni för typer av AI-tjänster?",
  "Vilka branscher jobbar ni med?",
  "Hur kontaktar man er, och var sitter ni?",
  "Har ni beskrivit hur ni jobbar med säkerhet/sekretess?",
  "Sammanfatta sidan 'Om oss' i tre punkter och länka källan.",
]

export function CompileitChat() {
  const [input, setInput] = useState("")

  const { messages, sendMessage, status, error, clearError } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
  })

  const isLoading = status === "submitted" || status === "streaming"
  const hasMessages = messages.length > 0

  const errorMessage = useMemo(() => {
    if (!error) {
      return undefined
    }

    return error.message || "Något gick fel när svaret skulle hämtas."
  }, [error])

  const submitPrompt = async (prompt: string) => {
    const trimmedPrompt = prompt.trim()
    if (!trimmedPrompt || isLoading) {
      return
    }

    clearError()
    setInput("")
    await sendMessage({ text: trimmedPrompt })
  }

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    void submitPrompt(input)
  }

  return (
    <div className="compileit-background relative isolate min-h-dvh overflow-hidden text-slate-100">
      <div className="relative z-10 flex min-h-dvh flex-col">
        <header className="border-b border-slate-900/90 bg-slate-950/80 backdrop-blur-sm">
          <div className="mx-auto flex h-14 w-full max-w-6xl items-center px-4 sm:px-6">
            <p className="font-mono text-sm text-slate-400">
              <span className="mr-2 text-blue-500">&gt;_</span>
              Compileits AI-assistent
            </p>
          </div>
        </header>

        <main className="mx-auto flex w-full max-w-6xl flex-1 px-3 sm:px-6">
          {hasMessages ? (
            <ChatMessageList messages={messages} status={status} />
          ) : (
            <section className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center pb-48 text-center sm:pb-56">
              <h1 className="text-4xl font-semibold tracking-tight text-slate-100 sm:text-5xl">
                Hej! Fråga mig om <span className="text-blue-400">Compileit</span>
              </h1>
              <p className="mt-4 max-w-xl text-balance text-base leading-7 text-slate-400 sm:text-xl">
                Jag kan svara på frågor om Compileits tjänster, kunder, case och mycket mer.
              </p>

              <ChatStarterPrompts
                prompts={STARTER_PROMPTS}
                disabled={isLoading}
                onSelectPrompt={(prompt) => {
                  void submitPrompt(prompt)
                }}
              />
            </section>
          )}
        </main>

        <ChatComposer
          input={input}
          isLoading={isLoading}
          errorMessage={errorMessage}
          onInputChange={(value) => {
            setInput(value)
            if (error) {
              clearError()
            }
          }}
          onSubmit={handleSubmit}
        />
      </div>
    </div>
  )
}
