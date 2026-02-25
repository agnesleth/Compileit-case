# Project overview
Use this guide to build the frontend for a web-based chat application. The app allows users to ask questions in Swedish about the company Compileit (compileit.com) and get answers from a custom RAG (Retrieval-Augmented Generation) backend. 

# Feature requirements
- Tech stack: Next.js (App Router), Tailwind CSS, shadcn/ui, and Vercel AI SDK (for streaming).
- Create a minimal, responsive chat interface. It should work perfectly on both mobile and desktop.
- The UI language MUST be in Swedish (e.g., input placeholder "Ställ en fråga om Compileit...").
- Build a scrollable message history area and a fixed input form at the bottom.
- Implement streaming for the AI responses to reduce perceived latency.
- Have a nice loading state or animation when the AI is processing/generating the answer.
- Style source links/citations nicely (e.g., using shadcn `Badge` components below the AI message).
- Distinguish visually between user messages and AI messages (e.g., different background colors, avatars).

# Relevant docs
## How to use Vercel AI SDK with a custom backend
```javascript
import { useChat } from 'ai/react';

export default function ChatComponent() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    // This will point to our Python/LangGraph backend route
    api: '/api/chat', 
  });
  
  // Use messages to map out the chat history
}

# Current file structure 
comp-case/
├── .next/ # Next.js build output (generated)
├── app/
│ ├── favicon.ico
│ ├── globals.css
│ ├── layout.tsx
│ └── page.tsx
├── components/
│ └── ui/
│ ├── avatar.tsx
│ ├── badge.tsx
│ ├── button.tsx
│ ├── card.tsx
│ ├── input.tsx
│ └── scroll-area.tsx
├── lib/
│ └── utils.ts
├── node_modules/ # Dependencies (generated)
├── public/
│ ├── file.svg
│ ├── globe.svg
│ ├── next.svg
│ ├── vercel.svg
│ └── window.svg
├── requirements/
│ └── frontend_instuctions.md
├── .gitignore
├── components.json # shadcn/ui config
├── eslint.config.mjs
├── next-env.d.ts
├── next.config.ts
├── package-lock.json
├── package.json
├── postcss.config.mjs
├── README.md
└── tsconfig.json

# Rules
All new custom components should go in /components and be named like example-component.tsx unless otherwise specified.

Use npx shadcn@latest add <component> when adding new base UI components. Do not write them from scratch.

All new pages go in /app.

Use Tailwind for all styling.