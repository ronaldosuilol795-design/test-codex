import { describe, it, expect } from "vitest";
import { clamp, unique, groupBy, sleep, invariant } from "../src/index.js";

describe("clamp", () => {
  it("should return the value when within range", () => {
    expect(clamp(5, 0, 10)).toBe(5);
  });

  it("should clamp to min when value is below", () => {
    expect(clamp(-1, 0, 10)).toBe(0);
  });

  it("should clamp to max when value is above", () => {
    expect(clamp(15, 0, 10)).toBe(10);
  });

  it("should handle equal min and max", () => {
    expect(clamp(5, 3, 3)).toBe(3);
  });
});

describe("unique", () => {
  it("should remove duplicate values", () => {
    expect(unique([1, 2, 2, 3, 3, 3])).toEqual([1, 2, 3]);
  });

  it("should preserve insertion order", () => {
    expect(unique([3, 1, 2, 1, 3])).toEqual([3, 1, 2]);
  });

  it("should return empty array for empty input", () => {
    expect(unique([])).toEqual([]);
  });

  it("should work with strings", () => {
    expect(unique(["a", "b", "a"])).toEqual(["a", "b"]);
  });
});

describe("groupBy", () => {
  it("should group items by key function", () => {
    const items = [
      { type: "fruit", name: "apple" },
      { type: "vegetable", name: "carrot" },
      { type: "fruit", name: "banana" },
    ];
    const grouped = groupBy(items, (i) => i.type);
    expect(grouped).toEqual({
      fruit: [
        { type: "fruit", name: "apple" },
        { type: "fruit", name: "banana" },
      ],
      vegetable: [{ type: "vegetable", name: "carrot" }],
    });
  });

  it("should return empty object for empty array", () => {
    expect(groupBy([], () => "key")).toEqual({});
  });
});

describe("sleep", () => {
  it("should resolve after the given duration", async () => {
    const start = Date.now();
    await sleep(50);
    const elapsed = Date.now() - start;
    expect(elapsed).toBeGreaterThanOrEqual(40);
  });
});

describe("invariant", () => {
  it("should not throw for truthy values", () => {
    expect(() => invariant(true)).not.toThrow();
    expect(() => invariant(1)).not.toThrow();
    expect(() => invariant("hello")).not.toThrow();
  });

  it("should throw for falsy values", () => {
    expect(() => invariant(false)).toThrow("Invariant violation");
    expect(() => invariant(null)).toThrow("Invariant violation");
    expect(() => invariant(0)).toThrow("Invariant violation");
    expect(() => invariant("")).toThrow("Invariant violation");
  });

  it("should throw with custom message", () => {
    expect(() => invariant(false, "custom error")).toThrow("custom error");
  });
});
