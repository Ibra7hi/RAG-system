"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Sparkles, Trash2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I'm your RAG Assistant. Ask me anything about your documents.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load chat messages from local cache (localStorage) on mount
  useEffect(() => {
    const cached = localStorage.getItem("rag_chat_messages");
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMessages(parsed);
        }
      } catch (e) {
        console.error("Error loading chat from local cache:", e);
      }
    }
  }, []);

  // Save chat messages to local cache (localStorage) when changed
  useEffect(() => {
    localStorage.setItem("rag_chat_messages", JSON.stringify(messages));
  }, [messages]);

  // Clear chat cache and reset messages
  const handleClearCache = () => {
    localStorage.removeItem("rag_chat_messages");
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Hello! I'm your RAG Assistant. Ask me anything about your documents.",
      },
    ]);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const apiUrl = "/api/chat";
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userMessage.content }),
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.response,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Error: ${error.message || "Failed to connect to the server."}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen items-center justify-center p-4 md:p-8">
      <div className="flex flex-col w-full max-w-4xl h-[90vh] bg-white rounded-3xl shadow-apple-floating overflow-hidden relative border border-gray-100">
        
        {/* Apple-style Glassmorphism Header */}
        <header className="absolute top-0 w-full z-10 px-6 py-4 bg-white/70 backdrop-blur-xl border-b border-gray-200/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500 rounded-xl text-white shadow-sm">
              <Sparkles size={20} />
            </div>
            <div>
              <h1 className="font-semibold text-gray-900 leading-tight">RAG Intelligence</h1>
              <p className="text-xs text-gray-500 font-medium">Powered by Llama 3.1 & Next.js</p>
            </div>
          </div>
          
          <button
            onClick={handleClearCache}
            title="Clear Chat Cache"
            className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-xl transition-all cursor-pointer"
          >
            <Trash2 size={18} />
          </button>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-6 pt-24 pb-32">
          <div className="flex flex-col gap-6 max-w-3xl mx-auto">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                  className={`flex items-end gap-2 ${
                    msg.role === "user" ? "flex-row-reverse" : "flex-row"
                  }`}
                >
                  <div
                    className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      msg.role === "user" ? "bg-gray-900 text-white" : "bg-gray-100 text-gray-600 border border-gray-200"
                    }`}
                  >
                    {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  
                  <div
                    className={`px-5 py-3.5 max-w-[80%] rounded-2xl shadow-sm leading-relaxed prose prose-sm ${
                      msg.role === "user"
                        ? "bg-apple-blue text-white rounded-br-sm"
                        : "bg-apple-bubble text-apple-dark rounded-bl-sm"
                    }`}
                  >
                    {/* Basic text rendering, you'd use react-markdown for rich MD */}
                    {(msg.content ?? "").split("\n").map((line, i) => (
                      <span key={i}>
                        {line}
                        <br />
                      </span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Loading Indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-end gap-2"
              >
                <div className="shrink-0 w-8 h-8 rounded-full bg-gray-100 border border-gray-200 text-gray-600 flex items-center justify-center">
                  <Bot size={16} />
                </div>
                <div className="bg-apple-bubble text-apple-dark px-5 py-4 rounded-2xl rounded-bl-sm flex items-center gap-1.5 shadow-sm">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="absolute bottom-0 w-full p-4 bg-gradient-to-t from-white via-white to-transparent">
          <form
            onSubmit={handleSubmit}
            className="max-w-3xl mx-auto flex items-center gap-2 bg-white border border-gray-200/80 p-1.5 rounded-full shadow-apple transition-all focus-within:shadow-md focus-within:border-apple-blue/50"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about your documents..."
              className="flex-1 bg-transparent px-4 py-2 outline-none text-gray-800 placeholder-gray-400"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="p-2.5 bg-apple-blue text-white rounded-full hover:bg-apple-blue-hover disabled:opacity-50 disabled:hover:bg-apple-blue transition-all active:scale-95"
            >
              <Send size={18} className="mr-0.5" />
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
