import { Users } from "lucide-react";
import PageHeader from "@/components/ui/PageHeader";
import EmptyState from "@/components/ui/EmptyState";

export default function TeamPage() {
  return (
    <div className="flex flex-col h-full">
      <PageHeader title="Team" />
      <EmptyState icon={Users} message="No team members yet." />
    </div>
  );
}
