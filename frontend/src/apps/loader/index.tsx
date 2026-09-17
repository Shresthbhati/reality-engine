'use client';

/**
 * LoaderApp — Reality Engine Evidence Ingestion Suite.
 * Houses both the Phone Companion loader and Computer Workstation loader.
 */

import React from 'react';
import { useREStore } from '@/store/re-store';
import { PhoneLoader } from './phone/PhoneLoader';
import { ComputerLoader } from './computer/ComputerLoader';

export default function LoaderApp() {
  const { loaderDeviceMode } = useREStore();

  if (loaderDeviceMode === 'phone') {
    return <PhoneLoader />;
  }

  return <ComputerLoader />;
}
