import React, { useEffect, useState, useRef } from 'react';

export function ChatPanel({ sessionId }: { sessionId: string }) {
    const [messages, setMessages] = useState<any[]>([]);
    const [input, setInput] = useState("");
    const [ws, setWs] = useState<WebSocket | null>(null);
    const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        fetch(`http://localhost:8000/chat/${sessionId}`)
            .then(res => res.json())
            .then(data => setMessages(data.history || []))
            .catch(() => { });

        const socket = new WebSocket(`ws://localhost:8000/ws/chat/${sessionId}`);
        socket.onopen = () => setWsStatus('connected');
        socket.onerror = () => setWsStatus('error');
        socket.onclose = () => setWsStatus('error');

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'chunk') {
                setIsTyping(true);
                setMessages(prev => {
                    const last = prev[prev.length - 1];
                    if (last && last.role === 'assistant' && !last.done) {
                        return [...prev.slice(0, -1), { ...last, content: last.content + data.text }];
                    }
                    return [...prev, { role: 'assistant', content: data.text, done: false }];
                });
            } else if (data.type === 'done') {
                setIsTyping(false);
                setMessages(prev => {
                    if (!prev.length) return prev;
                    return [...prev.slice(0, -1), { ...prev[prev.length - 1], done: true }];
                });
            }
        };

        setWs(socket);
        return () => socket.close();
    }, [sessionId]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const sendMessage = (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || !ws || ws.readyState !== WebSocket.OPEN) return;
        setMessages(prev => [...prev, { role: 'user', content: input, done: true }]);
        ws.send(JSON.stringify({ message: input }));
        setInput("");
    };

    const statusDot = {
        connecting: '#f59e0b',
        connected: '#22c55e',
        error: '#ef4444',
    }[wsStatus];

    return (
        <div className="flex flex-col h-full" style={{ background: 'white' }}>

            {/* Header */}
            <div className="p-4 border-b flex items-center space-x-2"
                style={{ background: 'linear-gradient(90deg, #1e3a8a, #1d4ed8)', borderColor: '#1d4ed8' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', background: statusDot, display: 'inline-block', flexShrink: 0 }} />
                <h3 className="font-bold text-white text-sm tracking-wide">Clinical AI Assistant</h3>
                {wsStatus === 'error' && (
                    <span className="ml-auto text-xs" style={{ color: 'rgba(255,255,255,0.6)' }}>Disconnected</span>
                )}
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3" style={{ background: '#f8faff' }}>
                {messages.length === 0 && (
                    <div className="text-center mt-8 space-y-2">
                        <div style={{ fontSize: 32 }}>💬</div>
                        <p className="text-sm" style={{ color: '#94a3b8' }}>
                            Ask anything about the scan results...
                        </p>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className="max-w-[88%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap leading-relaxed"
                            style={msg.role === 'user' ? {
                                background: 'linear-gradient(135deg, #1d4ed8, #3b82f6)',
                                color: 'white',
                                borderBottomRightRadius: 4,
                            } : {
                                background: 'white',
                                color: '#1e293b',
                                border: '1px solid #dbeafe',
                                borderBottomLeftRadius: 4,
                                boxShadow: '0 1px 4px rgba(30,58,138,0.07)'
                            }}>
                            {msg.content}
                            {msg.role === 'assistant' && !msg.done && (
                                <span style={{ color: '#1d4ed8' }} className="animate-pulse ml-1">▌</span>
                            )}
                        </div>
                    </div>
                ))}

                {isTyping && messages[messages.length - 1]?.role !== 'assistant' && (
                    <div className="flex justify-start">
                        <div className="rounded-2xl px-4 py-3 border"
                            style={{ background: 'white', borderColor: '#dbeafe' }}>
                            <span className="flex space-x-1">
                                {[0, 150, 300].map(delay => (
                                    <span key={delay}
                                        className="animate-bounce"
                                        style={{
                                            width: 7, height: 7, borderRadius: '50%',
                                            background: '#1d4ed8', display: 'inline-block',
                                            animationDelay: `${delay}ms`
                                        }} />
                                ))}
                            </span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t" style={{ background: 'white', borderColor: '#dbeafe' }}>
                {wsStatus === 'error' && (
                    <p className="text-xs mb-2 px-2 py-1 rounded"
                        style={{ color: '#b91c1c', background: '#fef2f2', border: '1px solid #fecaca' }}>
                        ⚠️ Disconnected — check backend &amp; GROQ_API_KEY
                    </p>
                )}
                <form onSubmit={sendMessage} className="flex space-x-2">
                    <input
                        type="text"
                        value={input}
                        onChange={e => setInput(e.target.value)}
                        placeholder="Ask about tumor metrics..."
                        disabled={wsStatus !== 'connected'}
                        className="flex-1 rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
                        style={{
                            border: '1.5px solid #bfdbfe',
                            background: wsStatus !== 'connected' ? '#f8faff' : 'white',
                            color: '#1e293b',
                        }}
                        onFocus={e => e.target.style.borderColor = '#1d4ed8'}
                        onBlur={e => e.target.style.borderColor = '#bfdbfe'}
                    />
                    <button
                        type="submit"
                        disabled={wsStatus !== 'connected' || !input.trim()}
                        className="px-4 py-2.5 rounded-xl font-semibold text-sm transition-all"
                        style={{
                            background: wsStatus === 'connected' && input.trim()
                                ? 'linear-gradient(135deg, #1d4ed8, #3b82f6)'
                                : '#e2e8f0',
                            color: wsStatus === 'connected' && input.trim() ? 'white' : '#94a3b8',
                            cursor: wsStatus !== 'connected' || !input.trim() ? 'not-allowed' : 'pointer',
                        }}>
                        Send
                    </button>
                </form>
            </div>
        </div>
    );
}