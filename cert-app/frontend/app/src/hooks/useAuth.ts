import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';
import type { User } from '@supabase/supabase-js';

const SESSION_DURATION_MS = 3 * 3600 * 1000; // hard 3-hour limit from login
const SESSION_START_KEY = 'auth_session_start';

export function useAuth() {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    const signOut = async () => {
        await supabase.auth.signOut();
        setUser(null);
        setToken(null);
        localStorage.removeItem(SESSION_START_KEY);
    };

    useEffect(() => {
        let cancelled = false;

        supabase.auth.getSession()
            .then(({ data: { session } }: any) => {
                if (cancelled) return;
                const currentUser = session?.user ?? null;
                setUser(currentUser);
                setToken(session?.access_token ?? null);
                if (session && !localStorage.getItem(SESSION_START_KEY)) {
                    localStorage.setItem(SESSION_START_KEY, Date.now().toString());
                }
                if (!session) localStorage.removeItem(SESSION_START_KEY);
                setLoading(false);
            })
            .catch((err: any) => {
                if (cancelled) return;
                const msg = err?.message ?? '';
                if (msg.includes('Refresh Token') || msg.includes('refresh_token') || msg.includes('Invalid Refresh Token')) {
                    supabase.auth.signOut({ scope: 'local' }).catch(() => {});
                    setUser(null);
                    setToken(null);
                }
                localStorage.removeItem(SESSION_START_KEY);
                setLoading(false);
            });

        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event: string, session: any) => {
            if (cancelled) return;
            const currentUser = session?.user ?? null;
            setUser(currentUser);
            setToken(session?.access_token ?? null);
            if (session) {
                if (!localStorage.getItem(SESSION_START_KEY)) {
                    localStorage.setItem(SESSION_START_KEY, Date.now().toString());
                }
            } else {
                localStorage.removeItem(SESSION_START_KEY);
            }
            setLoading(false);
        });

        return () => {
            cancelled = true;
            subscription.unsubscribe();
        };
    }, []);

    // Hard 3-hour session expiry — checked every minute, regardless of activity
    useEffect(() => {
        if (!user) return;

        const checkExpiry = () => {
            const startStr = localStorage.getItem(SESSION_START_KEY);
            if (!startStr) return;
            if (Date.now() - parseInt(startStr, 10) >= SESSION_DURATION_MS) {
                signOut();
            }
        };

        checkExpiry();
        const interval = setInterval(checkExpiry, 60 * 1000);
        return () => clearInterval(interval);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [user?.id]);

    return { user, token, loading, signOut };
}
