"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Globe,
  MoreVertical,
  Edit2,
  Copy,
  Archive,
  RotateCcw,
  Trash2,
  ExternalLink,
  AlertTriangle,
  Download,
} from "lucide-react";
import type { WorldRow } from "@/lib/types";
import {
  archiveWorld,
  deleteWorld,
  duplicateWorld,
  purgeWorld,
  renameWorld,
  restoreWorld,
} from "@/lib/api/worlds";

export interface WorldCardProps {
  world: WorldRow;
  statusView?: "active" | "archived" | "trash";
  onRefresh?: () => void;
}

export default function WorldCard({
  world,
  statusView = "active",
  onRefresh,
}: WorldCardProps) {
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [newName, setNewName] = useState(world.name);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const refresh = () => {
    onRefresh ? onRefresh() : router.refresh();
  };

  const handleRename = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim() || newName === world.name) {
      setRenaming(false);
      return;
    }
    setIsSubmitting(true);
    try {
      await renameWorld(world.id, newName.trim());
      setRenaming(false);
      refresh();
    } catch (err) {
      console.error("Rename failed", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDuplicate = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setMenuOpen(false);
    setIsSubmitting(true);
    try {
      await duplicateWorld(world.id);
      refresh();
    } catch (err) {
      console.error("Duplicate failed", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleArchive = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setMenuOpen(false);
    setIsSubmitting(true);
    try {
      await archiveWorld(world.id);
      refresh();
    } catch (err) {
      console.error("Archive failed", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRestore = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setMenuOpen(false);
    setIsSubmitting(true);
    try {
      await restoreWorld(world.id);
      refresh();
    } catch (err) {
      console.error("Restore failed", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async () => {
    setIsSubmitting(true);
    try {
      if (statusView === "trash") {
        await purgeWorld(world.id);
      } else {
        await deleteWorld(world.id);
      }
      setConfirmDelete(false);
      refresh();
    } catch (err) {
      console.error("Delete failed", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <div
        data-testid="world-card"
        data-world-id={world.id}
        className="relative flex flex-col rounded-lg overflow-hidden transition-colors group"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}
      >
        <Link
          href={`/worlds/${world.id}`}
          className="h-32 flex items-center justify-center shrink-0 block"
          style={{ background: "var(--bg-elevated)" }}
        >
          <Globe className="w-6 h-6" style={{ color: "var(--text-tertiary)" }} />
        </Link>

        {/* Menu Toggle Button */}
        <div className="absolute top-2 right-2 z-10">
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setMenuOpen(!menuOpen);
            }}
            data-testid="world-menu-btn"
            className="p-1 rounded-md bg-surface/70 hover:bg-elevated text-fg-muted transition"
            title="World options"
          >
            <MoreVertical className="w-4 h-4" />
          </button>

          {menuOpen && (
            <div
              className="absolute right-0 top-7 w-40 rounded-lg bg-surface border border-line shadow-xl py-1 text-xs text-fg z-20"
              onClick={(e) => e.stopPropagation()}
            >
              <Link
                href={`/worlds/${world.id}`}
                data-testid="world-open-btn"
                className="flex items-center gap-2 px-3 py-1.5 hover:bg-elevated transition"
                onClick={() => setMenuOpen(false)}
              >
                <ExternalLink className="w-3.5 h-3.5" />
                <span>Open World</span>
              </Link>

              {statusView === "active" && (
                <>
                  <Link
                    href={`/worlds/${world.id}?tab=export`}
                    data-testid="world-export-btn"
                    className="flex items-center gap-2 px-3 py-1.5 hover:bg-elevated transition"
                    onClick={() => setMenuOpen(false)}
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Export</span>
                  </Link>

                  <button
                    type="button"
                    data-testid="world-rename-btn"
                    onClick={() => {
                      setMenuOpen(false);
                      setRenaming(true);
                    }}
                    className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-elevated text-left transition"
                  >
                    <Edit2 className="w-3.5 h-3.5" />
                    <span>Rename</span>
                  </button>

                  <button
                    type="button"
                    data-testid="world-duplicate-btn"
                    onClick={handleDuplicate}
                    disabled={isSubmitting}
                    className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-elevated text-left transition"
                  >
                    <Copy className="w-3.5 h-3.5" />
                    <span>Duplicate</span>
                  </button>

                  <button
                    type="button"
                    data-testid="world-archive-btn"
                    onClick={handleArchive}
                    disabled={isSubmitting}
                    className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-elevated text-left transition"
                  >
                    <Archive className="w-3.5 h-3.5" />
                    <span>Archive</span>
                  </button>
                </>
              )}

              {(statusView === "archived" || statusView === "trash") && (
                <button
                  type="button"
                  onClick={handleRestore}
                  disabled={isSubmitting}
                  data-testid="world-restore-btn"
                  className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-elevated text-left text-success transition"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Restore</span>
                </button>
              )}

              <div className="my-1 border-t border-line-subtle" />

              <button
                type="button"
                onClick={() => {
                  setMenuOpen(false);
                  setConfirmDelete(true);
                }}
                data-testid="world-delete-btn"
                className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-error/40 text-left text-error transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>{statusView === "trash" ? "Permanently Delete" : "Delete"}</span>
              </button>
            </div>
          )}
        </div>

        {/* Card Body */}
        <div className="flex flex-col gap-1 p-4 flex-1">
          {renaming ? (
            <form onSubmit={handleRename} className="flex items-center gap-1">
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                autoFocus
                className="w-full bg-elevated border border-line-strong rounded px-2 py-0.5 text-xs text-fg focus:outline-none focus:border-accent"
              />
              <button
                type="submit"
                disabled={isSubmitting}
                className="px-2 py-0.5 bg-accent rounded text-[11px] text-on-primary"
              >
                Save
              </button>
              <button
                type="button"
                onClick={() => setRenaming(false)}
                className="px-1.5 py-0.5 hover:bg-line rounded text-[11px] text-fg-muted"
              >
                Cancel
              </button>
            </form>
          ) : (
            <Link
              href={`/worlds/${world.id}`}
              className="text-sm font-semibold hover:underline truncate"
              style={{ color: "var(--text-primary)" }}
            >
              {world.name}
            </Link>
          )}

          {world.location && (
            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {world.location}
            </span>
          )}

          <div
            className="flex items-center gap-3 mt-2 text-xs font-mono-num"
            style={{ color: "var(--text-secondary)" }}
          >
            {world.coverageKm2 != null && <span>{world.coverageKm2} km²</span>}
            <span>{world.sessionCount} Sessions</span>
            <span>{world.evidenceCount} Evidence</span>
          </div>

          {world.timeRangeStart && world.timeRangeEnd && (
            <span className="text-xs font-mono-num mt-1" style={{ color: "var(--text-tertiary)" }}>
              {world.timeRangeStart} – {world.timeRangeEnd}
            </span>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {confirmDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 select-none"
          onClick={() => setConfirmDelete(false)}
        >
          <div
            className="bg-surface border border-line rounded-xl max-w-md w-full p-5 space-y-4 shadow-2xl text-fg"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2.5 text-error font-semibold text-sm">
              <AlertTriangle className="w-5 h-5" />
              <span>
                {statusView === "trash" ? "Permanently delete world?" : "Delete world?"}
              </span>
            </div>

            <p className="text-xs text-fg-muted leading-relaxed">
              {statusView === "trash" ? (
                <>
                  Are you sure you want to permanently delete <strong>{world.name}</strong>?
                  All stored revisions, point clouds, meshes, and metadata will be permanently purged. This action cannot be undone.
                </>
              ) : (
                <>
                  Moving <strong>{world.name}</strong> to Trash will cancel any running jobs and archive current versions. You can restore it later from the Trash tab.
                </>
              )}
            </p>

            <div data-testid="delete-affected-items" className="rounded-md bg-elevated/60 border border-line p-3 text-xs space-y-1.5">
              <div className="font-medium text-fg">What will be {statusView === "trash" ? "permanently purged" : "moved to Trash"}:</div>
              <ul className="list-disc list-inside text-fg-muted space-y-0.5">
                <li>World: <span className="text-fg font-medium">{world.name}</span></li>
                <li>Sessions: <span className="text-fg font-medium">{world.sessionCount} session{world.sessionCount !== 1 ? "s" : ""}</span></li>
                <li>Evidence: <span className="text-fg font-medium">{world.evidenceCount} item{world.evidenceCount !== 1 ? "s" : ""}</span></li>
                {world.coverageKm2 != null && <li>Coverage: <span className="text-fg font-medium">{world.coverageKm2} km²</span></li>}
              </ul>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-line-subtle">
              <button
                type="button"
                onClick={() => setConfirmDelete(false)}
                data-testid="cancel-delete-btn"
                className="px-3 py-1.5 rounded-lg hover:bg-elevated text-xs text-fg-muted transition"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDelete}
                disabled={isSubmitting}
                data-testid="confirm-delete-btn"
                className="px-3 py-1.5 rounded-lg bg-error hover:bg-error text-on-error font-medium text-xs transition disabled:opacity-50"
              >
                {statusView === "trash" ? "Purge Permanently" : "Move to Trash"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
