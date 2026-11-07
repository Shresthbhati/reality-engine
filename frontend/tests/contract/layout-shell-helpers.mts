export function countEntities(world: { entities: Record<string, { semantic_labels?: string[] }> }, label: string): number {
  return Object.values(world.entities).filter((e) => (e.semantic_labels ?? []).includes(label)).length;
}
