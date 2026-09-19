import type { Metadata, Viewport } from 'next';
import MobileFieldApp from '@/apps/mobile/MobileFieldApp';

export const metadata: Metadata = {
  title: 'Reality Capture — Field',
  description: 'Reality Engine field capture: session, capture, evidence, sync.',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  themeColor: '#0d0e12',
};

export default function PhonePage() {
  return <MobileFieldApp />;
}
