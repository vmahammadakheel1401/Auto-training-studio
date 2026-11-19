import { buildDatasetFromCSV } from './csvParser';
import { LearningMode, ParsedDataset } from './types';

function createRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function normalNoise(rng: () => number, std = 1): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z * std;
}

export function generateHousePricesCSV(): string {
  const rng = createRng(4201);
  const neighborhoods = [
    { name: 'Downtown', premium: 95000 },
    { name: 'Oakridge', premium: 45000 },
    { name: 'Riverside', premium: 68000 },
    { name: 'Westside', premium: 25000 },
    { name: 'Hillcrest', premium: 120000 },
  ];
  const conditions = [
    { name: 'Fair', mult: 0.91 },
    { name: 'Good', mult: 1.0 },
    { name: 'Excellent', mult: 1.13 },
  ];

  const headers = [
    'property_id',
    'sqft_living',
    'bedrooms',
    'bathrooms',
    'lot_size_sqft',
    'year_built',
    'neighborhood',
    'condition',
    'garage_spaces',
    'sale_price',
  ];

  const lines = [headers.join(',')];

  for (let i = 1; i <= 180; i++) {
    const propertyId = `PROP-${String(1000 + i)}`;
    const sqft = Math.round(950 + rng() * 2650);
    const bedrooms = Math.min(6, Math.max(1, Math.round(sqft / 620 + normalNoise(rng, 0.45))));
    const bathrooms = Number(
      Math.min(4.5, Math.max(1, Math.round((bedrooms * 0.72 + rng() * 0.6) * 2) / 2)).toFixed(1)
    );
    const lotSize = Math.round(2800 + sqft * 1.9 + rng() * 4200);
    const yearBuilt = Math.round(1962 + rng() * 62);
    const nb = neighborhoods[Math.floor(rng() * neighborhoods.length)];
    const cond = conditions[Math.floor(rng() * conditions.length)];
    const garage = Math.min(3, Math.max(0, Math.round(sqft / 1200 + (rng() - 0.35))));

    const basePrice =
      115000 +
      sqft * 195 +
      bedrooms * 14500 +
      bathrooms * 22000 +
      lotSize * 6.5 +
      (yearBuilt - 1960) * 1650 +
      garage * 16000 +
      nb.premium;

    const salePrice = Math.round(
      (basePrice * cond.mult + normalNoise(rng, 18500)) / 500
    ) * 500;

    const lotOut = rng() < 0.025 ? '' : String(lotSize);
    const garageOut = rng() < 0.02 ? '' : String(garage);
    const condOut = rng() < 0.015 ? '' : cond.name;

    lines.push(
      [
        propertyId,
        sqft,
        bedrooms,
        bathrooms,
        lotOut,
        yearBuilt,
        nb.name,
        condOut,
        garageOut,
        Math.max(145000, salePrice),
      ].join(',')
    );
  }

  return lines.join('\n');
}

export function generateCustomerChurnCSV(): string {
  const rng = createRng(8817);
  const contracts = ['Month-to-Month', 'One Year', 'Two Year'];
  const payments = ['Electronic Check', 'Credit Card', 'Bank Transfer'];
  const internetServices = ['Fiber Optic', 'DSL', 'None'];

  const headers = [
    'customer_id',
    'tenure_months',
    'monthly_charges',
    'support_tickets',
    'contract_type',
    'payment_method',
    'internet_service',
    'paperless_billing',
    'churn',
  ];

  const lines = [headers.join(',')];

  for (let i = 1; i <= 200; i++) {
    const customerId = `CUST-${String(5000 + i)}`;
    const contract = contracts[Math.floor(rng() * contracts.length)];
    const tenureBase =
      contract === 'Two Year' ? 44 : contract === 'One Year' ? 28 : 12;
    const tenure = Math.min(
      72,
      Math.max(1, Math.round(tenureBase + normalNoise(rng, 14)))
    );

    const internet =
      internetServices[Math.floor(rng() * internetServices.length)];
    const baseCharge =
      internet === 'Fiber Optic' ? 88 : internet === 'DSL' ? 56 : 28;
    const monthlyCharges = Number(
      Math.max(20, Math.min(125, baseCharge + normalNoise(rng, 13))).toFixed(2)
    );

    const payment = payments[Math.floor(rng() * payments.length)];
    const paperless = rng() < 0.62 ? 'Yes' : 'No';

    let logit = -0.7;
    if (contract === 'Month-to-Month') logit += 1.45;
    else if (contract === 'Two Year') logit -= 1.35;
    else logit -= 0.45;

    logit -= (tenure - 24) * 0.045;
    logit += (monthlyCharges - 65) * 0.025;
    if (payment === 'Electronic Check') logit += 0.65;
    if (internet === 'Fiber Optic') logit += 0.45;
    if (paperless === 'Yes') logit += 0.25;

    const ticketLambda = logit > 0.2 ? 2.6 : 0.9;
    const supportTickets = Math.min(
      8,
      Math.max(0, Math.round(ticketLambda + Math.abs(normalNoise(rng, 1.1))))
    );
    logit += (supportTickets - 1.5) * 0.48;

    const prob = 1 / (1 + Math.exp(-logit));
    const churn = rng() < prob ? 'Yes' : 'No';

    const chargesOut = rng() < 0.02 ? '' : String(monthlyCharges);
    const paymentOut = rng() < 0.02 ? '' : payment;

    lines.push(
      [
        customerId,
        tenure,
        chargesOut,
        supportTickets,
        contract,
        paymentOut,
        internet,
        paperless,
        churn,
      ].join(',')
    );
  }

  return lines.join('\n');
}

export function generateCustomerSegmentationCSV(): string {
  const rng = createRng(9123);
  const headers = [
    'customer_id',
    'annual_income_k',
    'spending_score',
    'visit_frequency',
    'avg_basket_value',
    'tenure_years',
    'online_share_pct',
  ];

  // 4 realistic natural behavioral clusters (unlabeled)
  const prototypes = [
    { inc: 34, spend: 24, freq: 3, basket: 32, tenure: 2.1, online: 28 },
    { inc: 58, spend: 54, freq: 7, basket: 68, tenure: 4.8, online: 52 },
    { inc: 108, spend: 84, freq: 12, basket: 135, tenure: 6.4, online: 78 },
    { inc: 102, spend: 26, freq: 4, basket: 96, tenure: 5.5, online: 34 },
  ];

  const lines = [headers.join(',')];

  for (let i = 1; i <= 210; i++) {
    const cid = `SEG-${String(3000 + i)}`;
    const proto = prototypes[i % prototypes.length];

    const income = Math.max(18, Math.round(proto.inc + normalNoise(rng, 9)));
    const spending = Math.min(99, Math.max(4, Math.round(proto.spend + normalNoise(rng, 8))));
    const freq = Math.min(20, Math.max(1, Math.round(proto.freq + normalNoise(rng, 1.8))));
    const basket = Math.max(15, Math.round(proto.basket + normalNoise(rng, 14)));
    const tenure = Number(Math.max(0.5, proto.tenure + normalNoise(rng, 1.1)).toFixed(1));
    const online = Math.min(98, Math.max(5, Math.round(proto.online + normalNoise(rng, 10))));

    const tenureOut = rng() < 0.015 ? '' : String(tenure);

    lines.push([cid, income, spending, freq, basket, tenureOut, online].join(','));
  }

  return lines.join('\n');
}

export function generateMarketBasketCSV(): string {
  const rng = createRng(6541);
  const items = [
    'milk',
    'bread',
    'butter',
    'eggs',
    'coffee',
    'sugar',
    'cheese',
    'pasta',
    'tomato_sauce',
    'apples',
    'yogurt',
  ];

  const headers = ['transaction_id', ...items];
  const lines = [headers.join(',')];

  for (let i = 1; i <= 220; i++) {
    const tid = `TX-${String(7000 + i)}`;
    const basket = new Set<string>();

    // Morning breakfast archetype: bread + butter + eggs + milk
    if (rng() < 0.45) {
      basket.add('bread');
      if (rng() < 0.82) basket.add('butter');
      if (rng() < 0.72) basket.add('eggs');
      if (rng() < 0.68) basket.add('milk');
    }

    // Coffee lovers archetype: coffee + sugar + milk
    if (rng() < 0.42) {
      basket.add('coffee');
      if (rng() < 0.78) basket.add('sugar');
      if (rng() < 0.74) basket.add('milk');
    }

    // Italian dinner archetype: pasta + tomato_sauce + cheese
    if (rng() < 0.38) {
      basket.add('pasta');
      if (rng() < 0.86) basket.add('tomato_sauce');
      if (rng() < 0.75) basket.add('cheese');
    }

    // Healthy snack archetype: apples + yogurt
    if (rng() < 0.36) {
      basket.add('apples');
      if (rng() < 0.76) basket.add('yogurt');
    }

    // Occasional background items
    for (const item of items) {
      if (rng() < 0.08) basket.add(item);
    }

    // Guarantee non-empty basket
    if (basket.size === 0) {
      basket.add(items[Math.floor(rng() * items.length)]);
      basket.add(items[Math.floor(rng() * items.length)]);
    }

    const row = [tid, ...items.map((it) => (basket.has(it) ? '1' : '0'))];
    lines.push(row.join(','));
  }

  return lines.join('\n');
}

export type SampleDatasetKey =
  | 'house_prices'
  | 'customer_churn'
  | 'customer_segmentation'
  | 'market_basket';

export const SAMPLE_DATASET_META: Record<
  SampleDatasetKey,
  {
    key: SampleDatasetKey;
    title: string;
    dropdownLabel: string;
    fileName: string;
    learningMode: LearningMode;
    taskBadge: string;
    targetColumn: string;
    description: string;
    rowCountHint: number;
    colCountHint: number;
  }
> = {
  house_prices: {
    key: 'house_prices',
    title: 'House Prices',
    dropdownLabel: 'House Prices - Supervised Regression',
    fileName: 'house_prices.csv',
    learningMode: 'supervised',
    taskBadge: 'Supervised · Regression',
    targetColumn: 'sale_price',
    description:
      'Residential property listings with square footage, lot size, neighborhood, condition, and sale prices.',
    rowCountHint: 180,
    colCountHint: 10,
  },
  customer_churn: {
    key: 'customer_churn',
    title: 'Customer Churn',
    dropdownLabel: 'Customer Churn - Supervised Classification',
    fileName: 'customer_churn.csv',
    learningMode: 'supervised',
    taskBadge: 'Supervised · Classification',
    targetColumn: 'churn',
    description:
      'Subscription telemetry including tenure, contract terms, monthly charges, support tickets, and churn outcome.',
    rowCountHint: 200,
    colCountHint: 9,
  },
  customer_segmentation: {
    key: 'customer_segmentation',
    title: 'Customer Segmentation',
    dropdownLabel: 'Customer Segmentation - Unsupervised Clustering',
    fileName: 'customer_segmentation.csv',
    learningMode: 'unsupervised',
    taskBadge: 'Unsupervised · Unlabeled',
    targetColumn: '',
    description:
      'Unlabeled retail shopper profiles with income, spending score, visit frequency, basket value, and online share.',
    rowCountHint: 210,
    colCountHint: 7,
  },
  market_basket: {
    key: 'market_basket',
    title: 'Market Basket Transactions',
    dropdownLabel: 'Market Basket - Unsupervised Apriori',
    fileName: 'market_basket.csv',
    learningMode: 'unsupervised',
    taskBadge: 'Unsupervised · Basket Rules',
    targetColumn: '',
    description:
      'Unlabeled grocery checkout transactions across 11 binary item indicators for association rule mining.',
    rowCountHint: 220,
    colCountHint: 12,
  },
};

export function loadSampleDataset(key: SampleDatasetKey): ParsedDataset {
  const meta = SAMPLE_DATASET_META[key];
  let csv = '';
  if (key === 'house_prices') csv = generateHousePricesCSV();
  else if (key === 'customer_churn') csv = generateCustomerChurnCSV();
  else if (key === 'customer_segmentation') csv = generateCustomerSegmentationCSV();
  else csv = generateMarketBasketCSV();

  return buildDatasetFromCSV(
    csv,
    meta.fileName,
    meta.title,
    meta.targetColumn || undefined
  );
}
