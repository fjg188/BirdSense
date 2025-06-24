// Middleware to protect the /home route

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import {jwtVerify}  from 'jose';

const secret = new TextEncoder().encode(process.env.JWT_SECRET!);

async function isloggedIn(req: NextRequest) {
    const token = req.cookies.get('access')?.value;
    if (!token) return false;
    try {
        await jwtVerify(token, secret);
        return true;
    } catch(e) {
        return false;
    }
}

export async function middleware(req: NextRequest) {
    console.log('Middleware triggered for request:', req.url);
    const loggedIn = await isloggedIn(req);
    const url = req.nextUrl;

    if (url.pathname.startsWith('/home')){
        if(!loggedIn) {
            return NextResponse.redirect(new URL('/', req.url));
        }
    return NextResponse.next();
    }

    return NextResponse.next();
}

export const config = {
    matcher: ['/home/:path*', '/'],
};