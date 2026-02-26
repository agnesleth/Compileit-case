"use client"

import { useChat } from "@ai-sdk/react"
import { DefaultChatTransport } from "ai"
import { RotateCcw } from "lucide-react"
import type { FormEvent } from "react"
import { useMemo, useState } from "react"

import { ChatComposer } from "@/components/chat-composer"
import { ChatMessageList } from "@/components/chat-message-list"
import { ChatStarterPrompts } from "@/components/chat-starter-prompts"
import { Button } from "@/components/ui/button"

const STARTER_PROMPTS = [
  "Vad erbjuder ni för typer av AI-tjänster?",
  "Vilka branscher jobbar ni med?",
  "Hur kontaktar man er, och var sitter ni?",
  "Har ni beskrivit hur ni jobbar med säkerhet/sekretess?",
  "Sammanfatta sidan 'Om oss' i tre punkter och länka källan.",
]

function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID()
  }
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`
}

export function CompileitChat() {
  const [input, setInput] = useState("")
  const [chatId, setChatId] = useState<string>(() => createSessionId())

  const { messages, sendMessage, status, error, clearError, stop } = useChat({
    id: chatId,
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

  const handleResetChat = () => {
    stop()
    clearError()
    setInput("")
    setChatId(createSessionId())
  }

  return (
    <div className="compileit-background relative isolate min-h-dvh w-full overflow-hidden text-slate-100">
      <div className="relative z-10 flex min-h-dvh w-full min-w-0 flex-col">
        <header className="border-b border-slate-900/90 bg-slate-950/80 backdrop-blur-sm">
          <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between gap-3 px-4 sm:px-6">
            <p className="min-w-0 truncate font-mono text-sm text-slate-400">
              <span className="mr-2 text-blue-500">&gt;_</span>
              Compileits AI-assistent
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleResetChat}
              className="shrink-0 border-slate-700 bg-slate-900/70 text-slate-200 hover:bg-slate-800 hover:text-white"
              aria-label="Starta om chatten"
              title="Starta om chatten"
            >
              <RotateCcw className="size-4" />
              <span className="hidden sm:inline">Starta om</span>
            </Button>
          </div>
        </header>

        <main className="mx-auto flex w-full min-w-0 max-w-6xl flex-1 px-3 sm:px-6">
          {hasMessages ? (
            <ChatMessageList messages={messages} status={status} />
          ) : (
            <section className="mx-auto flex w-full min-w-0 max-w-3xl flex-1 flex-col items-center justify-center pb-44 text-center sm:pb-56">
              <h1 className="text-3xl font-semibold tracking-tight text-slate-100 sm:text-5xl">
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
