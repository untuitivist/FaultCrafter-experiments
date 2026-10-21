import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const options = new Map();
for (let index = 2; index < process.argv.length; index += 2) {
  options.set(process.argv[index], process.argv[index + 1]);
}
const run = path.resolve(options.get('--run') ?? path.join(project, 'outputs/comparisons_all_methods_5seeds_v3_20261020'));
const prepared = path.resolve(options.get('--data-run') ?? path.join(project, 'outputs/comparisons_v19_20261020'));
const runtime = path.resolve(options.get('--runtime') ?? path.join(project, 'tmp/experiment_workbook_20261020'));
const destination = path.resolve(options.get('--output') ?? path.join(run, 'comparison_experiments.xlsx'));

const devices = {
  hdu: 'HDU-BFDS', dirg: 'DIRG', hust: 'HUSTbearing', cwru: 'CWRU',
  bjtu: 'BJTU-RAO', hit_sq: 'HIT-SM SpectraQuest', hit_self: 'HIT-SM自建', pu: 'PU',
};
const methods = {
  real_only: 'Real-only', augmentation: '常规增强', mixstyle: 'MixStyle', coral: 'CORAL',
  superposition: '单故障波形叠加', diffucd: 'DiffUCD', ccgan: 'CcGAN',
  iccdm: 'iCCDM', time_weaver: 'Time Weaver', flowts: '条件FlowTS',
};
const channels = {
  motor: '电机', bearing_base: '轴承座', bearing_upper: '轴承平行上端',
  bearing_side: '轴承垂直侧端', drive_end: '驱动端', fan_end: '风扇端',
  vibration: '振动', vibration_1: '振动1',
};
const fields = [
  ['shaft_frequency_hz', '测试轴频（Hz）'], ['drive_frequency_hz', '测试驱动频率（Hz）'],
  ['radial_force_n', '测试径向力（N）'], ['lateral_force_n', '测试横向力（N）'],
  ['longitudinal_force_n', '测试纵向力（N）'], ['load_power_w', '测试负载功率（W）'],
  ['load_torque_nm', '测试负载扭矩（N·m）'],
];
const metricFields = [
  ['wasserstein', 'Wasserstein距离'], ['js', 'JS散度'], ['rms_error', 'RMS误差'],
  ['kurtosis_error', '峰度误差'], ['psd_error', '功率谱误差'], ['envelope_error', '包络谱误差'],
  ['embedding_mmd2', '嵌入MMD²'], ['content_mmd2', '内容MMD²'], ['style_mmd2', '风格MMD²'],
  ['frechet', 'Fréchet特征距离'], ['pairwise_distance', '样本成对距离'],
  ['nearest_source_distance', '最近源样本距离'], ['duplicate_rate', '重复率'],
  ['label_consistency', '故障标签一致率'], ['condition_response_error', '工况响应误差'],
];
const readJson = async filename => JSON.parse(await fs.readFile(filename, 'utf8'));
const unique = values => [...new Set(values)].sort();
const average = values => {
  const valid = values.filter(value => value !== null && value !== undefined);
  return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : null;
};
const channelName = value => channels[value] ?? value;
function compareResults(first, second) {
  const keys = job => [job.track, Object.keys(devices).indexOf(job.device), job.channel ?? '',
    job.target, Object.keys(methods).indexOf(job.method), job.arm ?? '', job.split];
  const left = keys(first.job), right = keys(second.job);
  for (let index = 0; index < left.length; index++) {
    if (left[index] < right[index]) return -1;
    if (left[index] > right[index]) return 1;
  }
  return 0;
}
async function loadResults(folder) {
  const directories = await fs.readdir(path.join(run, folder), { withFileTypes: true });
  const results = await Promise.all(directories.filter(entry => entry.isDirectory()).map(async entry => {
    const directory = path.join(run, folder, entry.name);
    try {
      return { ...await readJson(path.join(directory, 'result.json')),
        settings: await readJson(path.join(directory, 'job.json')) };
    } catch (error) {
      if (error.code === 'ENOENT') return null;
      throw error;
    }
  }));
  return results.filter(Boolean).sort(compareResults);
}

const [diagnostics, generators, inventory, ready, groups, quality, environment] = await Promise.all([
  loadResults('jobs'), loadResults('generators'), readJson(path.join(prepared, 'inventory.json')),
  readJson(path.join(prepared, 'data_ready.json')), readJson(path.join(run, 'diagnostic_groups.json')),
  readJson(path.join(run, 'quality_summary.json')), readJson(path.join(run, 'runtime.json')),
]);
const diagnosticIds = new Map(diagnostics.map((result, index) => [result.job.id, `D${String(index + 1).padStart(4, '0')}`]));
const generatorIds = new Map(generators.map((result, index) => [result.job.id, `G${String(index + 1).padStart(4, '0')}`]));
const groupResults = new Map(groups.records.map(result => [result.job.id, result]));
const conditions = new Map(inventory.records.map(record => [`${record.device}/${record.condition}`, record.coordinates]));
const schema = generators[0].schema;
const generatorSources = new Map();
for (const result of generators) {
  const manifest = await readJson(path.join(run, 'generators', result.job.id, 'source_manifest.json'));
  generatorSources.set(result.job.id, Object.fromEntries(['train', 'validation'].map(role => [role,
    manifest[role].map(({ samples, ...cell }) => ({ ...cell, sample_count: samples.length })),
  ])));
}
const physical = job => {
  const values = conditions.get(`${job.device}/${job.target}`);
  if (!values || values.length !== schema.device_field_indices[job.device].length) {
    throw new Error(`Missing physical coordinates for ${job.id}`);
  }
  return Object.fromEntries(schema.device_field_indices[job.device].map((field, index) => [schema.fields[field], values[index]]));
};
const coordinateText = (device, condition) => conditions.get(`${device}/${condition}`)
  .map(value => Number(value.toPrecision(10))).join(', ');
function trainingGroup(job) {
  if (job.track === 'T3') return job.method === 'real_only' ? '真实数据' : job.method === 'augmentation' ? '真实数据＋常规增强' : '真实数据＋多工况生成';
  if (job.method === 'augmentation') return '真实数据＋常规增强';
  if (job.method === 'superposition') return '真实数据＋波形叠加';
  if (job.arm === 'generated_only') return '仅生成数据';
  if (job.arm === 'real_generated' || generatorIds.has(job.parent)) return '真实数据＋生成数据';
  return '真实数据';
}
function identity(result, generator = false) {
  const job = result.job;
  return {
    experiment: (generator ? generatorIds : diagnosticIds).get(job.id), seed: job.split,
    task: job.track, device: devices[job.device], method: methods[job.method],
    channel: channelName(job.channel), training_group: generator ? null : trainingGroup(job),
    parent_experiment: generatorIds.get(job.parent) ?? null,
  };
}
function sourceFor(job) {
  const own = generators.find(result => result.job.device === job.device && result.job.target === job.target && result.job.split === job.split);
  if (!own) throw new Error(`No source manifest for ${job.id}`);
  return generatorSources.get(own.job.id);
}
function sourceSummary(manifest) {
  const grouped = selector => unique(manifest.train.map(cell => cell.device)).map(device =>
    `${devices[device]}: ${unique(manifest.train.filter(cell => cell.device === device).map(selector)).join('; ')}`,
  ).join('\n');
  return {
    training_devices: unique(manifest.train.map(cell => devices[cell.device])).join('; '),
    training_channels: grouped(cell => channelName(cell.channel)),
    training_conditions: grouped(cell => `(${coordinateText(cell.device, cell.condition)})`),
    training_states: grouped(cell => cell.state),
    training_windows: manifest.train.reduce((total, cell) => total + cell.sample_count, 0),
    validation_windows: manifest.validation.reduce((total, cell) => total + cell.sample_count, 0),
  };
}
function computation(result) {
  return {
    parameters: result.cost.parameters_total, trainable_parameters: result.cost.parameters_trainable,
    forward_gflops: result.cost.gflops_counted_per_sample, training_seconds: result.training_seconds,
    validation_seconds: result.validation_seconds, evaluation_seconds: result.evaluation_seconds ?? null,
    wall_seconds: result.wall_seconds, peak_gib: result.peak_cuda_bytes / 2 ** 30,
    peak_reserved_gib: result.peak_cuda_reserved_bytes == null ? null : result.peak_cuda_reserved_bytes / 2 ** 30,
    completed: result.completed, checkpoint_sha256: result.best_checkpoint_sha256,
  };
}
const generatorSettings = generators.map(result => ({
  ...identity(result, true), ...physical(result.job), ...sourceSummary(generatorSources.get(result.job.id)),
  native_length: result.settings.native_length, batch_size: result.settings.generator_batch_size,
  optimizer: result.settings.optimizer?.name ?? (result.settings.code['generators.py'] === '49afa21b1f9dfbd59c28bfa70a1f1c0d4c68e86ee7b995af483b0bafc7ec936b' ? 'Adam' : null),
  learning_rate: result.settings.optimizer?.learning_rate ?? (result.settings.code['generators.py'] === '49afa21b1f9dfbd59c28bfa70a1f1c0d4c68e86ee7b995af483b0bafc7ec936b' ? 0.0001 : null),
  updates_per_phase: result.settings.updates_per_phase, sampling_steps: result.settings.sampling_steps,
  max_epochs: result.settings.recipe?.max_epochs, microbatch_size: result.settings.microbatch_size,
  schedule: JSON.stringify(result.settings.recipe?.schedule ?? null),
  early_stopping: JSON.stringify(result.settings.recipe?.early_stopping ?? null),
  training_source: result.settings.recipe?.provenance,
  weight_decay: result.settings.optimizer?.weight_decay,
  sampling_nfe: result.sampling_nfe, precision: result.settings.precision, gpu: environment.gpu,
  implementation: result.settings.implementation, job_id: result.job.id, parent_job_id: result.job.parent,
  protocol_sha256: result.job.protocol_sha256, source_identity: result.source_identity,
  configuration_identity: result.identity,
}));
const generatorResults = generators.map(result => ({
  ...identity(result, true), ...computation(result), source_validation_loss: result.source_validation_loss,
  best_step: result.best_step, updates: result.updates_completed ?? result.history?.length,
  epochs_completed: result.epochs_completed, stop_reason: result.stop_reason, auxiliary_seconds: result.auxiliary_seconds,
  total_training_seconds: result.training_seconds + result.auxiliary_seconds,
  auxiliary_phase_updates: Object.entries(result.auxiliary_training).flatMap(([name, phases]) =>
    Object.entries(phases).map(([phase, history]) => `${name}/${phase}: ${history.updates_completed ?? history.length}`)).join('; '),
  sampling_steps: result.sampling_steps, sampling_nfe: result.sampling_nfe,
  sampler_gflops: result.sampler_cost.gflops_counted_per_sample,
  sampling_ms: result.sampling_seconds_batch_one * 1000, job_id: result.job.id,
  parent_sha256: result.parent_sha256,
}));
const diagnosticSettings = diagnostics.map(result => {
  const job = result.job, settings = result.settings;
  const manifest = sourceFor(job);
  const train = manifest.train.filter(cell => cell.device === job.device && cell.channel === job.channel);
  const validation = manifest.validation.filter(cell => cell.device === job.device && cell.channel === job.channel);
  const sourceCount = train.reduce((total, cell) => total + cell.sample_count, 0);
  if (sourceCount !== settings.source_windows) throw new Error(`Source count mismatch: ${job.id}`);
  const isAugmentation = job.method === 'augmentation';
  return {
    ...identity(result), ...physical(job),
    training_conditions: unique(train.map(cell => `(${coordinateText(cell.device, cell.condition)})`)).join('; '),
    training_condition_count: unique(train.map(cell => cell.condition)).length,
    training_states: unique(train.map(cell => cell.state)).join('; '),
    components: settings.components.join('; '), component_count: settings.components.length,
    available_source_windows: settings.source_windows,
    real_training_windows: job.arm === 'generated_only' ? 0 : settings.source_windows,
    augmented_training_windows: isAugmentation ? settings.additional_windows : 0,
    generated_training_windows: ['real_only', 'mixstyle', 'augmentation', 'coral'].includes(job.method) ? 0 : settings.additional_windows,
    validation_windows: validation.reduce((total, cell) => total + cell.sample_count, 0),
    real_windows_per_cell: ready.quotas.train, validation_windows_per_cell: ready.quotas.validation,
    target_windows: result.target.total, seen_windows: result.seen_conditions.total,
    native_length: result.cost.tensor_input_shapes[0].at(-1), batch_size: settings.batch_size,
    optimizer: settings.optimizer.name, learning_rate: settings.optimizer.learning_rate,
    weight_decay: settings.optimizer.weight_decay, max_epochs: settings.max_epochs,
    early_stopping: JSON.stringify(settings.early_stopping ?? null),
    epochs_completed: result.epochs_completed, best_epoch: result.best_epoch, stop_reason: result.stop_reason,
    loss: settings.loss, updates: settings.updates, formal_updates: settings.formal_updates,
    updates_per_epoch: settings.updates_per_epoch, arm: job.arm, gpu: environment.gpu,
    job_id: job.id, parent_job_id: job.parent, protocol_sha256: job.protocol_sha256,
    source_identity: result.source_identity, initial_weights_sha256: result.initial_weights_sha256,
    configuration_identity: result.identity,
  };
});
const diagnosticResults = diagnostics.map(result => {
  const groups = groupResults.get(result.job.id);
  const scores = {
    ...identity(result), ...computation(result), target_windows: result.target.total,
    target_correct: result.target.correct, target_acc: result.target.exact_set_accuracy,
    target_macro_f1: result.target.macro_f1_valid, target_macro_auc: average(result.target.auc_valid),
    target_unpredicted: result.target.unpredicted, seen_windows: result.seen_conditions.total,
    seen_acc: result.seen_conditions.exact_set_accuracy, seen_macro_f1: result.seen_conditions.macro_f1_valid,
    seen_macro_auc: average(result.seen_conditions.auc_valid), source_validation_acc: result.source_accuracy,
    best_step: result.best_step, updates: result.updates_completed, job_id: result.job.id,
    false_negative_rate: groups?.overall.false_negative_rate ?? null,
    false_positive_rate: groups?.overall.false_positive_rate ?? null,
  };
  for (const role of ['normal', 'single_fault', 'seen_composite', 'unseen_composite']) {
    const values = groups?.groups[role];
    scores[`${role}_windows`] = values?.total ?? null;
    scores[`${role}_acc`] = values?.exact_set_accuracy ?? null;
    scores[`${role}_macro_f1`] = values?.macro_f1_valid ?? null;
    scores[`${role}_macro_auc`] = values ? average(values.auc_valid) : null;
  }
  return scores;
});
const qualityResults = [...quality.records].sort(compareResults).map((entry, index) => {
  if (!diagnosticIds.has(entry.job.id) || !generatorIds.has(entry.job.parent)) throw new Error('Unlinked quality record');
  return {
    quality_experiment: `Q${String(index + 1).padStart(4, '0')}`, ...identity(entry), ...physical(entry.job),
    ...entry.metrics, state_count: entry.per_state.length,
    samples_per_distribution: entry.per_state.reduce((total, state) => total + state.count_per_distribution, 0),
    native_length: entry.window_length, scale: entry.scale, feature_precision: entry.feature_precision,
    feature_projection: entry.feature_projection, reference_cnn_sha256: entry.reference_cnn_sha256,
    generator_sha256: entry.generator_sha256, job_id: entry.job.id,
  };
});

const integer = '#,##0', decimal = '0.000', rate = '0.00%', scientific = '0.000E+00';
const column = (key, title, width = 18, format = null) => ({ key, title, width, format });
const commonColumns = [column('experiment', '实验编号', 12), column('seed', '种子', 9, integer),
  column('task', '任务', 9), column('device', '设备', 25), column('method', '方法', 19)];
const diagnosticColumns = [...commonColumns, column('channel', '通道', 21),
  column('training_group', '训练数据组', 29), column('parent_experiment', '生成器实验编号', 17)];
const generatorColumns = [...commonColumns, column('parent_experiment', '继承底座实验编号', 19)];
const physicalColumns = fields.map(([key, title]) => column(key, title, 22, decimal));
const identityColumns = [column('job_id', '运行ID', 28), column('protocol_sha256', '实验协议SHA256', 31),
  column('source_identity', '源数据身份SHA256', 31), column('configuration_identity', '运行配置身份SHA256', 31)];
const costColumns = [column('parameters', '参数总量（个）', 20, integer),
  column('trainable_parameters', '可训练参数量（个）', 23, integer),
  column('forward_gflops', '单次前向GFLOPs', 22, decimal), column('training_seconds', '主模型训练时间（s）', 24, decimal),
  column('validation_seconds', '验证时间（s）', 20, decimal), column('wall_seconds', '总耗时（s）', 20, decimal),
  column('peak_gib', '峰值显存（GiB）', 22, decimal)];
const sheets = [
  {
    name: '诊断结果', color: '#234C72', rows: diagnosticResults,
    columns: [...diagnosticColumns, column('target_windows', '测试样本量', 17, integer),
      column('target_correct', '测试正确数', 17, integer), column('target_acc', '测试ACC', 17, rate),
      column('target_macro_f1', '测试Macro-F1', 20, rate), column('target_macro_auc', '测试Macro-AUC', 21, rate),
      column('seen_windows', '已见工况测试样本量', 24, integer), column('seen_acc', '已见工况ACC', 20, rate),
      column('seen_macro_f1', '已见工况Macro-F1', 24, rate), column('seen_macro_auc', '已见工况Macro-AUC', 25, rate),
      ...[['normal', '正常'], ['single_fault', '单故障'], ['seen_composite', '已见复合故障'], ['unseen_composite', '未见复合故障']]
        .flatMap(([key, title]) => [column(`${key}_windows`, `${title}样本量`, 24, integer),
          column(`${key}_acc`, `${title}ACC`, 24, rate), column(`${key}_macro_f1`, `${title}Macro-F1`, 28, rate),
          column(`${key}_macro_auc`, `${title}Macro-AUC`, 29, rate)]),
      column('false_negative_rate', 'T4故障成分漏报率', 24, rate), column('false_positive_rate', 'T4故障成分误报率', 24, rate),
      column('source_validation_acc', '源验证集ACC', 20, rate), column('target_unpredicted', '未预测样本数', 19, integer),
      ...costColumns, column('evaluation_seconds', '评价时间（s）', 20, decimal),
      column('peak_reserved_gib', '峰值预留显存（GiB）', 25, decimal), column('updates', '实际更新次数', 19, integer),
      column('best_step', '最佳检查点步数', 21, integer), column('completed', '完成时间（UTC）', 33, 'yyyy-mm-dd hh:mm:ss.000'),
      column('job_id', '运行ID', 28), column('checkpoint_sha256', '模型权重SHA256', 31)],
  },
  {
    name: '生成质量', color: '#476889', rows: qualityResults,
    columns: [column('quality_experiment', '评价编号', 12), ...diagnosticColumns,
      ...metricFields.map(([key, title]) => column(key, title, 24,
        ['duplicate_rate', 'label_consistency'].includes(key) ? rate : decimal)),
      column('state_count', '故障状态数', 17, integer), column('samples_per_distribution', '每侧分布样本量', 21, integer),
      column('native_length', '信号长度（点）', 20, integer), ...physicalColumns,
      column('scale', '信号幅值尺度', 43), column('feature_precision', '特征计算精度', 19),
      column('feature_projection', '共同特征变换', 48), column('job_id', '关联诊断运行ID', 28),
      column('reference_cnn_sha256', '特征诊断器权重SHA256', 34), column('generator_sha256', '生成器权重SHA256', 31)],
  },
  {
    name: '生成器结果', color: '#345775', rows: generatorResults,
    columns: [...generatorColumns, column('source_validation_loss', '源验证损失', 20, decimal),
      ...costColumns, column('auxiliary_seconds', '辅助模块训练时间（s）', 28, decimal),
      column('total_training_seconds', '主模型与辅助训练合计（s）', 31, decimal),
      column('sampler_gflops', '完整采样GFLOPs', 23, decimal), column('sampling_steps', '采样步数', 16, integer),
      column('sampling_nfe', '实际采样NFE', 19, integer), column('sampling_ms', '单条生成时间（ms）', 25, decimal),
      column('updates', '主模型实际更新次数', 25, integer), column('best_step', '最佳检查点步数', 22, integer),
      column('epochs_completed', '实际完成轮数', 20, integer), column('stop_reason', '停止原因', 25),
      column('auxiliary_phase_updates', '各辅助阶段更新次数', 37), column('completed', '完成时间（UTC）', 33, 'yyyy-mm-dd hh:mm:ss.000'),
      column('job_id', '运行ID', 28), column('checkpoint_sha256', '模型权重SHA256', 31),
      column('parent_sha256', '继承底座权重SHA256', 33)],
  },
  {
    name: '诊断设置', color: '#879CB0', rows: diagnosticSettings,
    columns: [...diagnosticColumns, ...physicalColumns, column('training_condition_count', '源训练工况数', 20, integer),
      column('training_conditions', '源训练工况坐标', 47), column('training_states', '源训练故障状态', 35),
      column('components', '故障成分标签顺序', 42), column('component_count', '故障输出维数', 20, integer),
      column('available_source_windows', '可用真实训练样本量', 25, integer),
      column('real_training_windows', '使用真实训练样本量', 25, integer),
      column('augmented_training_windows', '常规增强样本量', 21, integer),
      column('generated_training_windows', '生成／叠加样本量', 23, integer),
      column('validation_windows', '真实验证样本量', 21, integer),
      column('real_windows_per_cell', '每个源工况与状态训练样本量', 33, integer),
      column('validation_windows_per_cell', '每个源工况与状态验证样本量', 33, integer),
      column('target_windows', '测试样本量', 17, integer), column('seen_windows', '已见工况测试样本量', 25, integer),
      column('native_length', '信号长度（点）', 21, integer), column('batch_size', '训练批量', 16, integer),
      column('optimizer', '优化器', 15), column('learning_rate', '学习率', 16, scientific), column('loss', '诊断损失函数', 26),
      column('weight_decay', '权重衰减', 16, scientific), column('max_epochs', '最大训练轮数', 20, integer),
      column('early_stopping', '早停规则', 52), column('epochs_completed', '实际完成轮数', 20, integer),
      column('best_epoch', '最佳模型轮数', 20, integer), column('stop_reason', '停止原因', 25),
      column('updates', '本次设置更新次数', 23, integer), column('updates_per_epoch', '每轮更新次数', 20, integer),
      column('formal_updates', '正式训练计划更新次数', 28, integer), column('arm', '训练组代码', 25),
      column('gpu', '运行GPU', 38), ...identityColumns, column('parent_job_id', '生成器运行ID', 28),
      column('initial_weights_sha256', '初始CNN权重SHA256', 32)],
  },
  {
    name: '生成器设置', color: '#9BAABA', rows: generatorSettings,
    columns: [...generatorColumns, ...physicalColumns.map(value => ({ ...value, title: value.title.replace('测试', '留出') })),
      column('training_devices', '训练设备', 42), column('training_channels', '训练通道', 52),
      column('training_conditions', '训练工况坐标', 58), column('training_states', '训练故障状态', 52),
      column('training_windows', '训练样本量', 19, integer), column('validation_windows', '验证样本量', 19, integer),
      column('native_length', '信号长度（点）', 21, integer), column('batch_size', '生成器训练批量', 22, integer),
      column('optimizer', '优化器', 15), column('learning_rate', '主模型学习率', 20, scientific),
      column('weight_decay', '权重衰减', 16, scientific), column('max_epochs', '最大训练轮数', 20, integer),
      column('microbatch_size', '显存分批大小', 20, integer), column('schedule', '学习率调度', 50),
      column('early_stopping', '早停规则', 52), column('training_source', '训练设置来源', 48),
      column('updates_per_phase', '主生成器最大更新次数', 29, integer), column('sampling_steps', '采样步数', 17, integer),
      column('sampling_nfe', '实际采样NFE', 20, integer), column('precision', '计算精度', 47), column('gpu', '运行GPU', 38),
      column('implementation', '实现方式', 36), ...identityColumns, column('parent_job_id', '继承底座运行ID', 28)],
  },
];

try {
  const link = await readJson(path.join(run, 'analysis.json'));
  const analysisDirectory = path.resolve(run, link.directory);
  const analysis = await readJson(path.join(analysisDirectory, 'summary.json'));
  const baseColumns = [column('experiment', '分析运行ID', 28), column('seed', '种子', 9, integer),
    column('device', '设备', 25), column('channel', '通道', 21), column('method', '方法／参考网络', 30)];
  const analysisIdentity = (entry, experiment) => ({ experiment, seed: entry.split ?? entry.job.split,
    device: devices[entry.device ?? entry.job.device], channel: channelName(entry.channel ?? entry.job.channel),
    method: entry.variant ?? methods[entry.job.method] });
  const controlledSettings = entry => ({
    mechanism: entry.mechanism ?? (entry.variant === 'ddpm' ? 'ddpm' : 'flow'),
    loss_space: entry.loss_space ?? (entry.variant === 'ddpm' ? 'epsilon' : 'v'),
    sampling_nfe: entry.sampling_nfe ?? entry.sampling_steps,
    comparisons: (entry.comparisons ?? (entry.variant === 'flow_v' ? ['prediction', 'mechanism', 'multichannel']
      : entry.variant === 'ddpm' ? ['mechanism']
        : ['flow_x', 'flow_epsilon'].includes(entry.variant) ? ['prediction'] : ['multichannel'])).join(', '),
  });
  const controlledRows = analysis.controlled.map(entry => ({ ...analysisIdentity(entry, entry.id),
    ...controlledSettings(entry),
    prediction: entry.prediction, group: entry.group, updates: entry.updates, sampling_steps: entry.sampling_steps,
    parameters: entry.cost.parameters_total, forward_gflops: entry.cost.gflops_counted_per_sample,
    sampler_gflops: entry.sampler_cost.gflops_counted_per_sample, training_seconds: entry.training_seconds,
    sampling_ms: entry.sampling_seconds_batch_one == null ? null : entry.sampling_seconds_batch_one * 1000,
    real_acc: entry.real_only.exact_set_accuracy, acc: entry.diagnostic.exact_set_accuracy,
    macro_f1: entry.diagnostic.macro_f1_valid, ...entry.metrics, source_identity: entry.source_identity,
    checkpoint_sha256: entry.best_checkpoint_sha256, initial_weights_sha256: entry.initial_weights_sha256,
    reference_cnn_sha256: entry.reference_cnn_sha256,
  }));
  sheets.push({ name: '受控比较', color: '#385B78', rows: controlledRows,
    columns: [...baseColumns, column('comparisons', '所属比较', 38), column('mechanism', '生成机制', 16),
      column('prediction', '直接预测目标', 20), column('loss_space', '损失空间', 16), column('group', '通道训练组', 21),
      column('real_acc', 'Real-only ACC', 22, rate), column('acc', 'R+G ACC', 19, rate),
      column('macro_f1', 'R+G Macro-F1', 22, rate), column('embedding_mmd2', '嵌入MMD²', 22, decimal),
      column('frechet', 'Fréchet距离', 22, decimal), column('training_seconds', '生成器训练时间（s）', 26, decimal),
      column('sampling_ms', '单条生成时间（ms）', 25, decimal),
      column('parameters', '生成器参数量', 23, integer), column('forward_gflops', '前向GFLOPs', 23, decimal),
      column('sampler_gflops', '采样GFLOPs', 23, decimal), column('updates', '更新次数', 17, integer),
      column('sampling_steps', '采样步数', 17, integer), column('sampling_nfe', '网络调用次数', 19, integer),
      column('source_identity', '训练数据身份', 32),
      column('initial_weights_sha256', '初始权重SHA256', 32), column('checkpoint_sha256', '最终权重SHA256', 32),
      column('reference_cnn_sha256', '特征CNN权重SHA256', 32)] });
  const scanRows = [], responseRows = [], compositeRows = [];
  const settingsRows = analysis.controlled.map(entry => ({ ...analysisIdentity(entry, entry.id),
    ...controlledSettings(entry),
    ...physical(entry), target: entry.target, network: entry.network, prediction: entry.prediction,
    group: entry.group, updates: entry.updates, sampling_steps: entry.sampling_steps, samples: 63,
    length: 1024, sample_rate: inventory.records.find(record => record.device === entry.device).sample_rate,
    scale: 'parent_normalized', source_identity: entry.source_identity,
    generator_sha256: entry.best_checkpoint_sha256, reference_cnn_sha256: entry.reference_cnn_sha256,
    arrays: path.join(analysisDirectory, 'controlled', entry.id, 'signals.npz') }));
  for (const entry of analysis.conditions) {
    const common = analysisIdentity(entry, entry.job.id);
    settingsRows.push({ ...common, ...physical(entry.job), target: entry.job.target, parent: entry.job.parent,
      sampling_steps: entry.sampling_steps, samples: entry.samples_per_request, length: entry.window_length,
      sample_rate: entry.sample_rate_hz, scale: entry.signal_scale, source_identity: entry.source_identity,
      generator_sha256: entry.generator_sha256,
      reference_cnn_sha256: entry.reference_cnn_sha256, arrays: path.join(analysisDirectory, 'conditions', entry.job.id, 'arrays.npz') });
    for (const cell of entry.per_cell) {
      const row = { ...common, condition: cell.condition, state: cell.state, drive_frequency_hz: cell.coordinates[0],
        lateral_force_n: cell.coordinates[1], longitudinal_force_n: cell.coordinates[2],
        unseen_composite: cell.unseen_composite ? '未见复合故障' : '已见故障状态',
        real_rms: cell.real_response[0], generated_rms: cell.generated_response[0],
        real_peak_hz: cell.real_response[1], generated_peak_hz: cell.generated_response[1],
        real_envelope_peak_hz: cell.real_response[2], generated_envelope_peak_hz: cell.generated_response[2],
        real_count: cell.count_real, generated_count: cell.count_generated, ...cell.metrics, array_key: cell.array_key };
      (cell.unseen_composite ? compositeRows : scanRows).push(row);
    }
    responseRows.push(...entry.responses.map(edge => ({ ...common, ...edge,
      axis_name: edge.axis === 0 ? '电驱动频率（Hz）' : '横向力（N）',
      real_rms_delta: edge.real_delta[0], generated_rms_delta: edge.generated_delta[0],
      real_peak_delta_hz: edge.real_delta[1], generated_peak_delta_hz: edge.generated_delta[1],
      real_envelope_delta_hz: edge.real_delta[2], generated_envelope_delta_hz: edge.generated_delta[2] })));
  }
  const scanColumns = [...baseColumns, column('condition', '工况代码', 24), column('state', '故障状态', 26),
    column('drive_frequency_hz', '电驱动频率（Hz）', 24, decimal), column('lateral_force_n', '横向力（N）', 21, decimal),
    column('longitudinal_force_n', '纵向力（N）', 21, decimal), column('unseen_composite', '故障训练状态', 23),
    column('real_count', '真实样本数', 19, integer), column('generated_count', '生成样本数', 19, integer),
    column('real_rms', '真实RMS', 22, decimal), column('generated_rms', '生成RMS', 22, decimal),
    column('real_peak_hz', '真实谱峰（Hz）', 25, decimal), column('generated_peak_hz', '生成谱峰（Hz）', 25, decimal),
    column('real_envelope_peak_hz', '真实包络谱峰（Hz）', 29, decimal), column('generated_envelope_peak_hz', '生成包络谱峰（Hz）', 29, decimal),
    ...metricFields.slice(0, 9).map(([key, title]) => column(key, title, 24, decimal)),
    column('label_consistency', '标签一致率', 23, rate), column('array_key', '原始数组索引', 23)];
  sheets.push({ name: '工况扫描', color: '#476889', rows: scanRows, columns: scanColumns });
  sheets.push({ name: '工况响应', color: '#577A88', rows: responseRows,
    columns: [...baseColumns, column('state', '故障状态', 25), column('arm', '请求方式', 23), column('axis_name', '变化参数', 26),
      column('first', '起始工况', 25), column('second', '终止工况', 25),
      ...[['real_rms_delta', '真实RMS变化'], ['generated_rms_delta', '生成RMS变化'], ['rms_response_error', 'RMS响应误差'],
        ['real_peak_delta_hz', '真实谱峰变化（Hz）'], ['generated_peak_delta_hz', '生成谱峰变化（Hz）'],
        ['peak_frequency_response_error', '谱峰响应误差（Hz）'], ['real_envelope_delta_hz', '真实包络谱峰变化（Hz）'],
        ['generated_envelope_delta_hz', '生成包络谱峰变化（Hz）'], ['envelope_peak_response_error', '包络谱峰响应误差（Hz）']]
        .map(([key, title]) => column(key, title, 31, decimal))] });
  sheets.push({ name: '复合故障分析', color: '#685B7E', rows: compositeRows,
    columns: [...scanColumns, column('interaction_error', '双相干均方误差', 27, decimal)] });
  sheets.push({ name: '分析设置', color: '#9BAABA', rows: settingsRows,
    columns: [...baseColumns, column('target', '留出工况', 25), ...physicalColumns,
      column('network', '受控参考网络', 37), column('comparisons', '所属比较', 38),
      column('mechanism', '受控生成机制', 22), column('prediction', '直接预测目标', 22),
      column('loss_space', '损失空间', 16), column('sampling_nfe', '受控网络调用次数', 23, integer),
      column('group', '受控通道训练组', 25), column('updates', '受控训练更新次数', 25, integer),
      column('parent', '冻结T2生成器运行ID', 30),
      column('sampling_steps', '采样步数', 18, integer), column('samples', '每个请求生成数', 23, integer),
      column('length', '原生信号长度（点）', 27, integer), column('sample_rate', '采样率（Hz）', 23, integer),
      column('scale', '信号尺度', 28), column('source_identity', '源数据身份SHA256', 34),
      column('generator_sha256', '冻结生成器权重SHA256', 34),
      column('reference_cnn_sha256', '特征CNN权重SHA256', 34), column('arrays', '波形及特征数组路径', 60)] });
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
}

if (options.has('--spec-only')) {
  await fs.writeFile(options.get('--spec-only'), JSON.stringify(sheets));
  process.exit(0);
}
const require = createRequire(path.join(runtime, 'runtime.cjs'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const workbook = Workbook.create();
const columnLetter = index => {
  let value = index + 1, label = '';
  while (value) {
    label = String.fromCharCode(65 + (value - 1) % 26) + label;
    value = Math.floor((value - 1) / 26);
  }
  return label;
};
await fs.mkdir(runtime, { recursive: true });
for (const [index, specification] of sheets.entries()) {
  const { columns, rows } = specification;
  const sheet = workbook.worksheets.add(specification.name);
  const lastColumn = columnLetter(columns.length - 1), lastRow = rows.length + 1;
  const matrix = [columns.map(value => value.title), ...rows.map(row => columns.map(value => row[value.key] ?? null))];
  sheet.getRange(`A1:${lastColumn}${lastRow}`).values = matrix;
  const body = sheet.getRange(`A1:${lastColumn}${lastRow}`);
  body.format.font = { name: 'Microsoft YaHei', size: 10, color: '#243447' };
  body.format.rowHeight = 23;
  body.format.verticalAlignment = 'center';
  body.format.wrapText = false;
  const table = sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, `ExperimentTable${index + 1}`);
  table.style = 'TableStyleMedium2';
  table.showFilterButton = true;
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format.fill = specification.color;
  header.format.font = { name: 'Microsoft YaHei', size: 10, bold: true, color: '#FFFFFF' };
  header.format.rowHeight = 42;
  header.format.wrapText = true;
  header.format.horizontalAlignment = 'center';
  header.format.borders = { insideVertical: { style: 'thin', color: '#FFFFFF' } };
  for (const [columnIndex, definition] of columns.entries()) {
    const letter = columnLetter(columnIndex);
    const range = sheet.getRange(`${letter}1:${letter}${lastRow}`);
    range.format.columnWidth = definition.width;
    const values = sheet.getRange(`${letter}2:${letter}${lastRow}`);
    values.format.horizontalAlignment = definition.format ? 'right' : 'left';
    if (['seed', 'task'].includes(definition.key)) values.format.horizontalAlignment = 'center';
    if (['training_devices', 'training_channels', 'training_conditions', 'training_states', 'components',
      'feature_projection', 'precision', 'auxiliary_phase_updates'].includes(definition.key)) values.format.wrapText = true;
    if (definition.format) values.setNumberFormat(definition.format);
  }
  sheet.getRange(`A2:${lastColumn}${lastRow}`).format.autofitRows();
  sheet.showGridLines = false;
  sheet.tabColor = specification.color;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(index === 1 ? 3 : 2);
  const seedColumn = columns.findIndex(value => value.key === 'seed');
  const seedValues = sheet.getRange(`${columnLetter(seedColumn)}2:${columnLetter(seedColumn)}${lastRow}`).values.flat();
  if (seedValues.some(seed => ![42, 43, 44, 45, 46].includes(seed))) throw new Error(`Invalid seed column: ${sheet.name}`);
  console.log(`${sheet.name}: ${rows.length} rows, ${columns.length} columns`);
}
workbook.recalculate();
console.log((await workbook.inspect({ kind: 'sheet', include: 'id,name', maxChars: 1800 })).ndjson);
for (const specification of sheets) {
  for (const [slice, range] of [['identity', 'A1:J9'], ['details', specification.name === '诊断结果' ? 'K1:V9'
    : specification.name === '生成质量' ? 'J1:U9' : specification.name === '生成器结果' ? 'G1:Q9'
      : specification.name === '诊断设置' ? 'V1:AF9' : specification.name === '生成器设置' ? 'R1:AB9' : 'K1:R9']]) {
    const preview = await workbook.render({ sheetName: specification.name, range, scale: 1.4, format: 'png' });
    await fs.writeFile(path.join(runtime, `${specification.name}_${slice}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
await fs.mkdir(path.dirname(destination), { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(destination);
await fs.writeFile(path.join(runtime, 'workbook_expected.json'), JSON.stringify(sheets.map(specification => ({
  name: specification.name, columns: specification.columns, rows: specification.rows.map(row =>
    specification.columns.map(column => row[column.key] ?? null)),
})), null, 2));
console.log(`Saved ${destination}`);
