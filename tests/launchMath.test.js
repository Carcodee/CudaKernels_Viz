const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {
  sanitizeConfig,
  computeLaunch,
  buildBlockList,
  isWithinThreadBudget,
  MAX_TOTAL_THREADS,
  WARP_SIZE,
  getThreadLinearIndex,
  getWarpId,
  getWarpsPerBlock,
  GPU_PROFILES,
  computeOccupancy,
  computeSmPacking,
  computeKernelUsageLimits,
  validateLaunchConfig,
  computeGpuResidency,
  recommendVisualizationBudget,
  computeAdaptiveVisualizationBudget,
  snapToPowerOfTwo,
  powerOfTwoExponent
} = require("../src/launchMath.js");

test("computeLaunch keeps explicit grid and block sizes without implicit division", function () {
  const launch = computeLaunch({
    grid: { x: 3, y: 2, z: 4 },
    block: { x: 8, y: 4, z: 2 }
  });

  assert.deepEqual(launch.grid, { x: 3, y: 2, z: 4 });
  assert.deepEqual(launch.block, { x: 8, y: 4, z: 2 });
  assert.equal(launch.threadsPerBlock, 64);
  assert.equal(launch.totalBlocks, 24);
  assert.equal(launch.totalThreadsLaunched, 1536);
});

test("sanitizeConfig clamps unsupported values", function () {
  const safe = sanitizeConfig({
    grid: { x: -10, y: 90, z: 0 },
    block: { x: 0, y: 4096, z: 99 }
  });

  assert.deepEqual(safe.grid, { x: 1, y: 32, z: 1 });
  assert.deepEqual(safe.block, { x: 1, y: 1024, z: 64 });
});

test("power-of-two helpers snap values for slider-based block controls", function () {
  assert.equal(snapToPowerOfTwo(3, 1024), 4);
  assert.equal(snapToPowerOfTwo(63, 64), 64);
  assert.equal(snapToPowerOfTwo(600, 1024), 512);
  assert.equal(powerOfTwoExponent(1, 1024), 0);
  assert.equal(powerOfTwoExponent(8, 1024), 3);
  assert.equal(powerOfTwoExponent(600, 1024), 9);
});

test("buildBlockList enumerates explicit grid coordinates", function () {
  const launch = computeLaunch({
    grid: { x: 2, y: 3, z: 2 },
    block: { x: 4, y: 3, z: 2 }
  });

  const blocks = buildBlockList(launch);
  const last = blocks[blocks.length - 1];

  assert.equal(blocks.length, 12);
  assert.deepEqual(last.block, { x: 1, y: 2, z: 1 });
  assert.deepEqual(last.threadCount, 24);
});

test("thread budget helper rejects launches above 500000 threads", function () {
  assert.equal(MAX_TOTAL_THREADS, 1000000);
  assert.equal(isWithinThreadBudget({
    grid: { x: 20, y: 20, z: 20 },
    block: { x: 5, y: 5, z: 5 }
  }), true);
  assert.equal(isWithinThreadBudget({
    grid: { x: 20, y: 20, z: 20 },
    block: { x: 5, y: 5, z: 6 }
  }), false);
});

test("warp mapping uses contiguous linear thread order with warp size 32", function () {
  assert.equal(WARP_SIZE, 32);
  assert.equal(getThreadLinearIndex({ x: 0, y: 0, z: 0 }, { x: 8, y: 4, z: 2 }), 0);
  assert.equal(getThreadLinearIndex({ x: 7, y: 3, z: 0 }, { x: 8, y: 4, z: 2 }), 31);
  assert.equal(getThreadLinearIndex({ x: 0, y: 0, z: 1 }, { x: 8, y: 4, z: 2 }), 32);
  assert.equal(getWarpId({ x: 7, y: 3, z: 0 }, { x: 8, y: 4, z: 2 }), 0);
  assert.equal(getWarpId({ x: 0, y: 0, z: 1 }, { x: 8, y: 4, z: 2 }), 1);
  assert.equal(getWarpsPerBlock({ x: 8, y: 4, z: 2 }), 2);
});

test("computeOccupancy respects SM resource limits for a selected GPU", function () {
  const occupancy = computeOccupancy(
    {
      block: { x: 16, y: 8, z: 1 },
      registersPerThread: 37,
      sharedMemoryPerBlockKB: 8
    },
    GPU_PROFILES.H100
  );

  assert.equal(occupancy.threadsPerBlock, 128);
  assert.equal(occupancy.warpsPerBlock, 4);
  assert.equal(occupancy.limiters.maxThreadsPerSM, 16);
  assert.equal(occupancy.limiters.maxWarpsPerSM, 16);
  assert.equal(occupancy.limiters.maxBlocksPerSM, 32);
  assert.equal(occupancy.activeBlocksPerSM, 13);
  assert.equal(occupancy.activeWarpsPerSM, 52);
  assert.equal(occupancy.occupancyRatio, 52 / 64);
});

test("computeSmPacking groups blocks into streaming multiprocessors by active block capacity", function () {
  const launch = computeLaunch({
    grid: { x: 3, y: 2, z: 2 },
    block: { x: 16, y: 8, z: 1 }
  });
  const packing = computeSmPacking(
    launch,
    {
      registersPerThread: 37,
      sharedMemoryPerBlockKB: 8
    },
    GPU_PROFILES.H100
  );

  assert.equal(packing.activeBlocksPerSM, 13);
  assert.equal(packing.sms.length, 1);
  assert.equal(packing.sms[0].blocks.length, 12);
  assert.equal(packing.sms[0].sharedMemoryKB, 96);
});

test("computeKernelUsageLimits derives slider caps from the selected GPU and block size", function () {
  const h100LargeBlock = computeKernelUsageLimits(
    { x: 32, y: 32, z: 1 },
    GPU_PROFILES.H100
  );
  const h100SmallBlock = computeKernelUsageLimits(
    { x: 16, y: 8, z: 1 },
    GPU_PROFILES.H100
  );

  assert.equal(h100LargeBlock.maxRegistersPerThread, 64);
  assert.equal(h100LargeBlock.maxSharedMemoryPerBlockKB, 228);
  assert.equal(h100SmallBlock.maxRegistersPerThread, 255);
});

test("validateLaunchConfig reports invalid block dimensions against the selected GPU", function () {
  const invalid = validateLaunchConfig(
    {
      grid: { x: 2, y: 1, z: 1 },
      block: { x: 2048, y: 1, z: 1 },
      threadsPerBlock: 2048
    },
    GPU_PROFILES.H100
  );

  assert.equal(invalid.isValid, false);
  assert.equal(invalid.violations[0].code, "block_dim_x");
});

test("computeGpuResidency separates resident blocks from queued blocks and waves", function () {
  const launch = computeLaunch({
    grid: { x: 20, y: 10, z: 1 },
    block: { x: 16, y: 8, z: 1 }
  });
  const residency = computeGpuResidency(
    launch,
    {
      registersPerThread: 37,
      sharedMemoryPerBlockKB: 8
    },
    GPU_PROFILES.H100
  );

  assert.equal(residency.blocksPerWave, 1560);
  assert.equal(residency.residentBlockCount, 200);
  assert.equal(residency.queuedBlockCount, 0);
  assert.equal(residency.waveCount, 1);
});

test("computeGpuResidency queues blocks when the launch exceeds whole-gpu resident capacity", function () {
  const launch = computeLaunch({
    grid: { x: 32, y: 32, z: 2 },
    block: { x: 16, y: 8, z: 1 }
  });
  const residency = computeGpuResidency(
    launch,
    {
      registersPerThread: 37,
      sharedMemoryPerBlockKB: 8
    },
    GPU_PROFILES.H100
  );

  assert.equal(residency.blocksPerWave, 1560);
  assert.equal(residency.residentBlockCount, 1560);
  assert.equal(residency.queuedBlockCount, 488);
  assert.equal(residency.waveCount, 2);
  assert.equal(residency.residentSMCount, 120);
});

test("recommendVisualizationBudget picks a safe fallback for cpu-only systems", function () {
  const budget = recommendVisualizationBudget({
    webglAvailable: false,
    hardwareConcurrency: 8
  });

  assert.equal(budget.maxThreads, 16000);
  assert.equal(budget.tier, "cpu");
});

test("recommendVisualizationBudget raises limits for integrated and discrete gpus", function () {
  const integrated = recommendVisualizationBudget({
    webglAvailable: true,
    renderer: "Intel(R) Iris(R) Xe Graphics",
    vendor: "Intel",
    hardwareConcurrency: 8
  });
  const discrete = recommendVisualizationBudget({
    webglAvailable: true,
    renderer: "NVIDIA GeForce RTX 3070",
    vendor: "NVIDIA",
    hardwareConcurrency: 16
  });
  const highEnd = recommendVisualizationBudget({
    webglAvailable: true,
    renderer: "NVIDIA GeForce RTX 4090",
    vendor: "NVIDIA",
    hardwareConcurrency: 24
  });

  assert.equal(integrated.maxThreads, 96000);
  assert.equal(integrated.tier, "integrated");
  assert.equal(discrete.maxThreads, 384000);
  assert.equal(discrete.tier, "discrete");
  assert.equal(highEnd.maxThreads, 1000000);
  assert.equal(highEnd.tier, "high");
});

test("computeAdaptiveVisualizationBudget only reduces when frame time is really slow", function () {
  const steady = computeAdaptiveVisualizationBudget({
    currentMaxThreads: 220000,
    baselineMaxThreads: 220000,
    averageFrameMs: 42
  });
  const slow = computeAdaptiveVisualizationBudget({
    currentMaxThreads: 220000,
    baselineMaxThreads: 220000,
    averageFrameMs: 185
  });

  assert.equal(steady.shouldReduce, false);
  assert.equal(steady.nextMaxThreads, 220000);
  assert.equal(slow.shouldReduce, true);
  assert.equal(slow.nextMaxThreads, 109984);
});

test("hero copy and viewport reserve layout space to avoid jumpy height changes", function () {
  const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
  const css = fs.readFileSync(path.join(__dirname, "..", "styles.css"), "utf8");
  const js = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");

  assert.match(html, /class="view-card launch-view-card"/);
  assert.match(html, /class="viewport-row"/);
  assert.match(html, /class="view-card viewport-card"/);
  assert.match(html, /class="view-card viewport-controls-card"/);
  assert.match(html, /id="combined-viewport-shell" class="viewport-shell"/);
  assert.match(html, /class="hero-copy"/);
  assert.match(html, /class="launch-info-stack"/);
  assert.match(html, /id="combined-info" class="view-note launch-info-primary"/);
  assert.match(html, /id="controls-hint" class="view-note launch-info-secondary"/);
  assert.doesNotMatch(html, /viewport-debug/);
  assert.match(css, /\.app-shell\s*\{[\s\S]*min-height:\s*calc\(100dvh\s*-\s*64px\);/);
  assert.match(css, /\.workspace\s*\{[\s\S]*min-height:\s*0;/);
  assert.match(css, /\.views\s*\{[\s\S]*gap:\s*20px;[\s\S]*min-height:\s*0;/);
  assert.match(css, /\.launch-view-card,\s*\.viewport-card,\s*\.viewport-controls-card\s*\{[\s\S]*display:\s*grid;[\s\S]*gap:\s*12px;/);
  assert.match(css, /\.viewport-row\s*\{[\s\S]*grid-template-columns:\s*minmax\(0, 2fr\) minmax\(280px, 1fr\);/);
  assert.match(css, /\.viewport-controls-card\s*\{[\s\S]*align-content:\s*start;/);
  assert.match(css, /\.viewport-shell\s*\{[\s\S]*min-height:\s*640px;[\s\S]*height:\s*100%;/);
  assert.match(css, /\.three-view\s*\{[\s\S]*min-height:\s*600px;/);
  assert.match(css, /\.launch-info-stack\s*\{[\s\S]*grid-template-rows:\s*72px 44px;/);
  assert.match(css, /\.launch-info-primary\s*\{[\s\S]*min-height:\s*72px;[\s\S]*max-height:\s*72px;/);
  assert.match(css, /\.launch-info-secondary\s*\{[\s\S]*min-height:\s*44px;[\s\S]*max-height:\s*44px;/);
  assert.match(css, /\.viewport-controls-grid\s*\{[\s\S]*grid-template-columns:\s*1fr;/);
  assert.match(js, /typeof ResizeObserver !== "undefined"/);
  assert.match(js, /resizeObserver\.observe\(combinedViewportShellRoot \|\| root\.parentElement \|\| root\)/);
  assert.doesNotMatch(js, /viewportDebugRoot/);
  assert.doesNotMatch(js, /updateViewportDebug/);
  assert.doesNotMatch(js, /root\.style\.height\s*=\s*Math\.max\(320, availableHeight\) \+ "px"/);
  assert.doesNotMatch(js, /root\.style\.height\s*=\s*""/);
  assert.match(css, /\.three-view canvas\s*\{[\s\S]*height:\s*100%;/);
  assert.match(css, /--surface-raised:\s*rgba\(246, 248, 247, 0\.84\);/);
  assert.match(css, /--shadow-soft:\s*0 18px 40px rgba\(17, 24, 39, 0\.12\);/);
  assert.match(css, /body\s*\{[\s\S]*font-family:\s*"Cascadia Code", "Cascadia Mono", "SFMono-Regular", "Consolas", monospace;/);
  assert.match(css, /h1,\s*h2\s*\{[\s\S]*font-family:\s*"Cascadia Code", "Cascadia Mono", "SFMono-Regular", "Consolas", monospace;/);
  assert.match(css, /\.summary-card,\s*\.controls,\s*\.view-card\s*\{[\s\S]*transition:\s*transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;/);
  assert.match(css, /\.field input\[type="range"\]\s*\{[\s\S]*appearance:\s*none;/);
  assert.match(css, /\.field input\[type="range"\]::\-webkit\-slider\-thumb\s*\{/);
  assert.match(css, /\.toolbar-button,\s*\.toolbar-toggle\s*\{[\s\S]*transition:\s*transform 160ms ease, background-color 160ms ease, border-color 160ms ease, color 160ms ease;/);
  assert.match(css, /@keyframes rise-in\s*\{/);
  assert.match(css, /@keyframes hero-fade\s*\{/);
  assert.match(css, /@keyframes viewport-breathe\s*\{/);
  assert.match(css, /@keyframes viewport-sheen\s*\{/);
  assert.match(css, /\.hero-copy\s*\{[\s\S]*animation:\s*hero-fade 560ms ease both;/);
  assert.match(css, /\.controls,\s*\.view-card\s*\{[\s\S]*animation:\s*rise-in 480ms ease both;/);
  assert.match(css, /\.three-view::before\s*\{[\s\S]*animation:\s*viewport-sheen 10s ease-in-out infinite;/);
  assert.match(css, /\.three-view\.is-hovered\s*\{[\s\S]*transform:\s*translateY\(-3px\) scale\(1\.002\);/);
  assert.match(css, /\.three-view\.is-dragging\s*\{[\s\S]*transform:\s*scale\(0\.9995\);/);
  assert.match(css, /\.three-view\.is-dragging canvas\s*\{[\s\S]*cursor:\s*grabbing;/);
  assert.match(css, /\.three-view\.is-hovered canvas\s*\{[\s\S]*transform:\s*scale\(1\.012\);/);
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)\s*\{[\s\S]*animation:\s*none !important;/);
  assert.match(js, /renderer\.domElement\.classList\.add\("scene-canvas"\)/);
  assert.match(js, /function syncInteractionState\(\)/);
  assert.match(js, /renderRadius:\s*12/);
  assert.match(js, /orbit\.renderTarget\.lerp\(orbit\.target, easing\)/);
  assert.match(js, /function animateScene\(timeSeconds\)/);
  assert.match(js, /function getSceneLayoutSignature\(\)/);
  assert.match(js, /state\._lastSceneLayoutSignature !== sceneLayoutSignature/);
  assert.match(js, /fitCameraToScene\(combinedHost, 2\.15\)/);
  assert.match(js, /function registerSpawn\(node, options\)/);
  assert.match(js, /function animateSpawnAnimations\(frameAt\)/);
  assert.match(js, /animationState\.spawnAnimations/);
  assert.match(js, /addSpawnedObject\(host, host\.rootGroup, shellMesh/);
  assert.match(js, /window\.requestAnimationFrame\(tick\)/);
  assert.match(js, /hoverArtifacts/);
  assert.match(js, /root\.classList\.toggle\("is-hovered"/);
  assert.match(js, /root\.classList\.toggle\("is-dragging"/);
  assert.match(js, /shellWire/);
  assert.match(js, /emissiveIntensity:\s*0\.42/);
  assert.match(js, /shellHighlight\.material\.opacity\s*=\s*thisHost\.hoverArtifacts\.baseShellOpacity/);
  assert.match(js, /threadHighlight\.material\.emissiveIntensity\s*=\s*0\.42 \+ Math\.sin\(timeSeconds \* 8\.4\) \* 0\.08/);
  assert.match(js, /materialEntry\.material\.opacity\s*=\s*materialEntry\.targetOpacity \* \(0\.18 \+ \(0\.82 \* eased\)\)/);
  assert.match(js, /var popStrength = Math\.sin\(progress \* Math\.PI\) \* 0\.12/);
  assert.match(js, /var scaleFactor = 0\.94 \+ \(0\.06 \* eased\) \+ popStrength/);
});
































