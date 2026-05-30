/**
 * test-codex — A systematic collection of TypeScript utilities.
 *
 * @packageDocumentation
 */

/**
 * Clamp `value` between `min` and `max` (inclusive).
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Return a new array with duplicate values removed, preserving insertion order.
 */
export function unique<T>(arr: readonly T[]): T[] {
  return [...new Set(arr)];
}

/**
 * Group an array of items by the value returned from `keyFn`.
 */
export function groupBy<T, K extends string | number | symbol>(
  arr: readonly T[],
  keyFn: (item: T) => K,
): Record<K, T[]> {
  const result = {} as Record<K, T[]>;
  for (const item of arr) {
    const key = keyFn(item);
    (result[key] ??= []).push(item);
  }
  return result;
}

/**
 * Pause execution for `ms` milliseconds.
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Invariant assertion — throws if `condition` is falsy.
 */
export function invariant(
  condition: unknown,
  message?: string,
): asserts condition {
  if (!condition) {
    throw new Error(message ?? "Invariant violation");
  }
}
