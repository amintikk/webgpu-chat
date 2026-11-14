// Conversations live only in this browser, in IndexedDB. Nothing is ever sent anywhere.
import type { GenStats } from '../engine/protocol.ts';

export interface StoredMessage {
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string;
  model?: string;
  stats?: GenStats;
  /** The answer was cut short (stopped by the user or by the token limit). */
  partial?: boolean;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  /** The title was written by the user (never overwritten automatically). */
  customTitle?: boolean;
  createdAt: number;
  updatedAt: number;
  model: string;
  messages: StoredMessage[];
}

const DB = 'webgpu-chat';
const STORE = 'conversations';

let dbp: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  dbp ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      const s = req.result.createObjectStore(STORE, { keyPath: 'id' });
      s.createIndex('updatedAt', 'updatedAt');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbp;
}

async function tx<T>(mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T> | void): Promise<T | undefined> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const r = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(r ? r.result : undefined);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error);
  });
}

export async function listConversations(): Promise<Conversation[]> {
  const all = (await tx<Conversation[]>('readonly', (s) => s.getAll())) ?? [];
  return all.sort((a, b) => b.updatedAt - a.updatedAt);
}

export const saveConversation = (c: Conversation) => tx('readwrite', (s) => s.put(c));
export const deleteConversation = (id: string) => tx('readwrite', (s) => s.delete(id));
export const clearConversations = () => tx('readwrite', (s) => s.clear());

export function newId(): string {
  return crypto.randomUUID();
}

/** First words of the first message, as a provisional title. */
export function titleFrom(text: string): string {
  const t = text.replace(/\s+/g, ' ').trim();
  return t.length > 60 ? `${t.slice(0, 57).trimEnd()}…` : t || '…';
}
