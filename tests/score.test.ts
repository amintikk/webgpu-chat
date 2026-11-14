import assert from 'node:assert/strict';
import { test } from 'node:test';
import { level, score, wait } from '../src/app/score.ts';

test('score is 0-10 and grows with speed', () => {
  assert.equal(score(0), 0);
  assert.equal(score(2), 0);
  assert.equal(score(60), 10);
  assert.equal(score(500), 10);
  assert.ok(score(10) > score(5) && score(20) > score(10));
});

test('verdicts follow reading speed', () => {
  assert.equal(level(3), 'slow');
  assert.equal(level(6), 'fair');
  assert.equal(level(12), 'good');
  assert.equal(level(25), 'great');
  assert.equal(level(50), 'excellent');
  assert.equal(wait(400), 'instant');
  assert.equal(wait(1500), 'quick');
  assert.equal(wait(3000), 'noticeable');
  assert.equal(wait(9000), 'long');
});
