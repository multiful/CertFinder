import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { toast } from 'sonner';

const COMPARE_KEY = 'certfinder-compare';
const MAX_COMPARE = 3;

export interface CompareItem {
  id: number;
  name: string;
}

interface CompareContextType {
  items: CompareItem[];
  addToCompare: (item: CompareItem) => void;
  removeFromCompare: (id: number) => void;
  clearCompare: () => void;
  isInCompare: (id: number) => boolean;
  canAdd: boolean;
}

const CompareContext = createContext<CompareContextType | null>(null);

export function CompareProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<CompareItem[]>(() => {
    try {
      const raw = localStorage.getItem(COMPARE_KEY);
      return raw ? (JSON.parse(raw) as CompareItem[]) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(COMPARE_KEY, JSON.stringify(items));
    } catch {}
  }, [items]);

  const addToCompare = useCallback((item: CompareItem) => {
    setItems(prev => {
      if (prev.some(i => i.id === item.id)) {
        toast.info('이미 비교 목록에 추가된 자격증입니다.');
        return prev;
      }
      if (prev.length >= MAX_COMPARE) {
        toast.error('최대 3개까지 비교할 수 있습니다.');
        return prev;
      }
      toast.success(`"${item.name}"을 비교 목록에 추가했습니다.`);
      return [...prev, item];
    });
  }, []);

  const removeFromCompare = useCallback((id: number) => {
    setItems(prev => prev.filter(i => i.id !== id));
  }, []);

  const clearCompare = useCallback(() => {
    setItems([]);
  }, []);

  const isInCompare = useCallback(
    (id: number) => items.some(i => i.id === id),
    [items]
  );

  return (
    <CompareContext.Provider
      value={{ items, addToCompare, removeFromCompare, clearCompare, isInCompare, canAdd: items.length < MAX_COMPARE }}
    >
      {children}
    </CompareContext.Provider>
  );
}

export function useCompare() {
  const ctx = useContext(CompareContext);
  if (!ctx) throw new Error('useCompare must be used within CompareProvider');
  return ctx;
}
