import { useState, useEffect, useRef } from 'react';
import { supabase } from '@/lib/supabase';
import type { User } from '@supabase/supabase-js';

// Inactivity timeout in milliseconds (1 hour = 3600000ms)
const INACTIVITY_TIMEOUT = 3600 * 1000;

export function useAuth() {
    const [user, setUser] = useState<User | null>(null);
    const [token, setToken] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    // Effect 1: Handle Authentication Session
    useEffect(() => {
        let cancelled = false;

        supabase.auth.getSession()
            .then(({ data: { session } }: any) => {
                if (cancelled) return;
                const currentUser = session?.user ?? null;
                setUser(currentUser);
                setToken(session?.access_token ?? null);
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
                setLoading(false);
            });

        const { data: { subscription } } = supabase.auth.onAuthStateChange((_event: string, session: any) => {
            if (cancelled) return;
            const currentUser = session?.user ?? null;
            setUser(currentUser);
            setToken(session?.access_token ?? null);
            setLoading(false);
        });

        return () => {
            cancelled = true;
            subscription.unsubscribe();
        };
    }, []); // Run only on mount

    // Effect 2: Handle Inactivity Timer when user is authenticated
    useEffect(() => {
        if (!user) return;

        const activityEvents = ['mousedown', 'mousemove', 'keydown', 'scroll', 'touchstart'];

        const handleActivity = () => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
            timeoutRef.current = setTimeout(() => {
                console.warn('Inactivity timeout reached. Signing out...');
                signOut();
            }, INACTIVITY_TIMEOUT);
        };

        activityEvents.forEach(event => {
            window.addEventListener(event, handleActivity);
        });

        // Initialize timer
        handleActivity();

        return () => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
            activityEvents.forEach(event => {
                window.removeEventListener(event, handleActivity);
            });
        };
    }, [user?.id]); // Depend on user ID instead of object reference

    const signOut = async () => {
        await supabase.auth.signOut();
        setUser(null);
        setToken(null);
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };

    return { user, token, loading, signOut };
}
