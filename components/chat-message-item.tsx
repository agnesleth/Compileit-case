"use client"

import type { UIMessage } from "ai"
import { Bot, UserRound } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { extractSourcesFromMessage } from "@/lib/chat-sources"
import { cn } from "@/lib/utils"

function getMessageText(message: UIMessage): string {
  const textFromParts = message.parts
    .filter((part): part is Extract<UIMessage["parts"][number], { type: "text" }> => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim()

  if (textFromParts) {
    return textFromParts
  }

  const fallbackContent = (message as unknown as { content?: unknown }).content
  if (typeof fallbackContent === "string") {
    return fallbackContent.trim()
  }

  return ""
}

function renderTextWithBoldMarkdown(text: string) {
  const segments = text.split(/(\*\*[^*\n]+?\*\*)/g)
  return segments.map((segment, index) => {
    const isBoldToken =
      segment.startsWith("**") && segment.endsWith("**") && segment.length > 4

    if (isBoldToken) {
      return (
        <strong key={`bold-${index}`} className="font-semibold">
          {segment.slice(2, -2)}
        </strong>
      )
    }

    return <span key={`text-${index}`}>{segment}</span>
  })
}

export function AssistantTypingIndicator() {
  return (
    <div className="flex w-full min-w-0 justify-start gap-3">
      <Avatar size="sm" className="mt-1 border border-blue-900/40 bg-slate-900/80">
        <AvatarFallback className="bg-transparent text-blue-300">
          <Bot className="size-3.5" />
        </AvatarFallback>
      </Avatar>
      <div className="min-w-0 max-w-[85%] rounded-2xl border border-slate-800/90 bg-slate-900/80 px-4 py-3 text-sm">
        <div className="flex items-center gap-1.5">
          <span className="size-2 animate-bounce rounded-full bg-blue-400 [animation-delay:-0.3s]" />
          <span className="size-2 animate-bounce rounded-full bg-blue-400/85 [animation-delay:-0.15s]" />
          <span className="size-2 animate-bounce rounded-full bg-blue-400/70" />
        </div>
      </div>
    </div>
  )
}

export function ChatMessageItem({ message }: { message: UIMessage }) {
  const isUser = message.role === "user"
  const text = getMessageText(message)
  const sources = isUser ? [] : extractSourcesFromMessage(message)

  return (
    <div className={cn("flex w-full min-w-0 gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <Avatar size="sm" className="mt-1 border border-blue-900/40 bg-slate-900/80">
          <AvatarFallback className="bg-transparent text-blue-300">
            <Bot className="size-3.5" />
          </AvatarFallback>
        </Avatar>
      )}

      <div className={cn("min-w-0 max-w-[85%] space-y-2", isUser ? "items-end" : "items-start")}>
        <div
          className={cn(
            "min-w-0 rounded-2xl border px-4 py-3 text-sm leading-7 whitespace-pre-wrap break-words [overflow-wrap:anywhere]",
            isUser
              ? "border-blue-500/20 bg-blue-600/20 text-blue-50"
              : "border-slate-800/90 bg-slate-900/80 text-slate-100"
          )}
        >
          {text ? renderTextWithBoldMarkdown(text) : "…"}
        </div>

        {!isUser && sources.length > 0 && (
          <div className="flex flex-wrap gap-2 pl-1">
            {sources.map((source) => (
              <Badge
                asChild
                variant="outline"
                className="max-w-full border-blue-500/30 bg-blue-500/10 text-blue-100 whitespace-normal break-words hover:bg-blue-500/20"
                key={`${source.url}-${source.title}`}
              >
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="block max-w-full break-words [overflow-wrap:anywhere]"
                >
                  {source.title}
                </a>
              </Badge>
            ))}
          </div>
        )}
      </div>

      {isUser && (
        <Avatar size="sm" className="mt-1 border border-blue-500/40 bg-blue-600/20">
          <AvatarFallback className="bg-transparent text-blue-100">
            <UserRound className="size-3.5" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  )
}
