//proxy route for login API 
//handlles JWT cookie 
import { NextRequest, NextResponse } from 'next/server';
import jwt from 'jsonwebtoken';

const secret = process.env.JWT_SECRET!;

export async function POST(req: NextRequest) {

    const { password } = await req.json();
    
    const lambda = await fetch(process.env.login_api_url!,{
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ password }),
    });

    if (!lambda.ok){
        return NextResponse.json({ ok: false}, { status: 401 });
    } 

    const token = await jwt.sign({ pass: password }, secret, { expiresIn: '7d' }); // token with 7 days expiration

    const res = NextResponse.json({ ok: true });
    res.cookies.set({
    name: 'access',
    value: token,
    httpOnly: true,                    
    sameSite: 'lax',
    path: '/',
    maxAge: 60 * 60 * 24 * 7,          // 7 days
    secure: process.env.NODE_ENV === 'production'
  });
  return res;
}