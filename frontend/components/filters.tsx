"use client";

import { Search, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export interface FilterState {
  riskLevel: string;
  trend: string;
  cohort: string;
  search: string;
}

export const EMPTY_FILTERS: FilterState = {
  riskLevel: "all",
  trend: "all",
  cohort: "all",
  search: "",
};

export function Filters({
  value,
  onChange,
  resultCount,
  totalCount,
}: {
  value: FilterState;
  onChange: (next: FilterState) => void;
  resultCount: number;
  totalCount: number;
}) {
  const set = (patch: Partial<FilterState>) => onChange({ ...value, ...patch });
  const dirty =
    value.riskLevel !== "all" ||
    value.trend !== "all" ||
    value.cohort !== "all" ||
    value.search !== "";

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="relative min-w-[14rem] flex-1">
        <Search
          className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
          aria-hidden
        />
        <Input
          value={value.search}
          onChange={(e) => set({ search: e.target.value })}
          placeholder="Search machine ID"
          aria-label="Search machine ID"
          className="pl-8"
        />
      </div>

      <div className="w-[10.5rem]">
        <Select value={value.riskLevel} onValueChange={(v) => set({ riskLevel: v })}>
          <SelectTrigger aria-label="Filter by risk level">
            <SelectValue placeholder="Risk level" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All risk levels</SelectItem>
            <SelectItem value="CRITICAL">Critical</SelectItem>
            <SelectItem value="HIGH">High</SelectItem>
            <SelectItem value="MEDIUM">Medium</SelectItem>
            <SelectItem value="LOW">Low</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="w-[10.5rem]">
        <Select value={value.trend} onValueChange={(v) => set({ trend: v })}>
          <SelectTrigger aria-label="Filter by risk trend">
            <SelectValue placeholder="Risk trend" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">Any trend</SelectItem>
            <SelectItem value="RISING">Rising</SelectItem>
            <SelectItem value="STABLE">Stable</SelectItem>
            <SelectItem value="FALLING">Falling</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="w-[11.5rem]">
        <Select value={value.cohort} onValueChange={(v) => set({ cohort: v })}>
          <SelectTrigger aria-label="Filter by machine age">
            <SelectValue placeholder="Machine age" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All machines</SelectItem>
            <SelectItem value="established">Established only</SelectItem>
            <SelectItem value="new">Newly commissioned</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <p className="text-sm text-muted-foreground">
        {resultCount} of {totalCount}
      </p>

      {dirty && (
        <Button variant="ghost" size="sm" onClick={() => onChange(EMPTY_FILTERS)}>
          <X className="h-3.5 w-3.5" aria-hidden />
          Clear
        </Button>
      )}
    </div>
  );
}
