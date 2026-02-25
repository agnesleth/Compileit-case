"use client"

import type { ChatStatus, UIMessage } from "ai"
import { useEffect, useRef } from "react"

import { ChatMessageItem, AssistantTypingIndicator } from "@/components/chat-message-item"
import { ScrollArea } from "@/components/ui/scroll-area"

export function ChatMessageList({
  messages,
  status,
}: {
  messages: UIMessage[]
  status: ChatStatus
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      block: "end",
      behavior: status === "ready" ? "smooth" : "auto",
    })
  }, [messages, status])

  const showTypingIndicator =
    (status === "submitted" || status === "streaming") &&
    messages[messages.length - 1]?.role === "user"

  return (
    <ScrollArea className="h-full w-full">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-0 pt-8 pb-48 sm:pt-10 sm:pb-56">
        {messages.map((message) => (
          <ChatMessageItem key={message.id} message={message} />
        ))}

        {showTypingIndicator && <AssistantTypingIndicator />}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  )
}
