use core::ops::Range;
use rand::prelude::{thread_rng, ThreadRng};
use rand::Rng;
use sarsa::LimitedList;
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::{
    cmp::{max, min},
    sync::atomic::{AtomicUsize, Ordering},
};

const M: usize = 20;
const N: usize = 20;
const DIMS: usize = 8;
const TAKE_LAST_N: usize = 100;

const TRIALS: usize = 100_000_000;
const MAX_STEPS: usize = 100;
const EPSILON: f64 = 0.12;
const GAMMA: f64 = 0.999;
const LEARNING_RATE: f64 = 0.47;

const THREAD_COUNT: usize = 20;
const REPEATS: usize = 20;

struct Coord {
    x: usize,
    y: usize,
}

impl Coord {
    fn random(m: Range<usize>, n: Range<usize>, rng: &mut ThreadRng) -> Coord {
        Coord {
            x: rng.gen_range(m),
            y: rng.gen_range(n),
        }
    }

    fn apply_action(&self, action: usize) -> Coord {
        let dx = [0, 1, 0, -1, 1, 1, -1, -1];
        let dy = [1, 0, -1, 0, 1, -1, 1, -1];

        Coord {
            x: max(0, min(19, self.x as i64 + dx[action])) as usize,
            y: max(0, min(19, self.y as i64 + dy[action])) as usize,
        }
    }
}

type QValues<const Z: usize> = [f64; Z];
type QGrid<const X: usize, const Y: usize, const Z: usize> = [[QValues<Z>; Y]; X];

fn rand_grid<const X: usize, const Y: usize, const Z: usize>(
    rng: &mut ThreadRng,
) -> QGrid<X, Y, Z> {
    let mut res = [[[0f64; Z]; Y]; X];

    for x in res.iter_mut() {
        for y in x.iter_mut() {
            for z in y.iter_mut() {
                *z = rng.gen();
            }
        }
    }

    res
}

fn choose_action<const Z: usize>(q: &QValues<Z>, rng: &mut ThreadRng) -> usize {
    if rng.gen::<f64>() > EPSILON {
        q.iter()
            .enumerate()
            .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
            .unwrap()
            .0
    } else {
        rng.gen_range(0..Z)
    }
}

fn reward(state: &Coord) -> u64 {
    match state {
        Coord { x: 1, y: 1 } => 25,
        Coord { x: 1, y: 18 } => 20,
        Coord { x: 18, y: 18 } => 15,
        Coord { x: 18, y: 1 } => 10,
        Coord { x: 10, y: 10 } => 5,
        _ => 0,
    }
}

fn sarsa<const LIMIT: usize>() -> LimitedList<u64, LIMIT> {
    let mut rng = thread_rng();
    let mut q = rand_grid::<M, N, DIMS>(&mut rng);
    let mut res = LimitedList::new();

    for _ in 0..TRIALS {
        let mut steps = 0;
        let mut s = Coord::random(0..M, 0..N, &mut rng);
        let mut a = choose_action(&q[s.x][s.y], &mut rng);
        let mut r = 0;

        while r == 0 && steps < MAX_STEPS {
            let s_n = s.apply_action(a);
            let a_n = choose_action(&q[s_n.x][s_n.y], &mut rng);

            r = reward(&s_n);
            q[s.x][s.y][a] +=
                LEARNING_RATE * (r as f64 + GAMMA * q[s_n.x][s_n.y][a_n] - q[s.x][s.y][a]);
            s = s_n;
            a = a_n;
            steps += 1;
        }

        res.push(r);
    }

    res.clean();
    res
}

fn main() {
    let actual_repeats = (REPEATS / THREAD_COUNT) * THREAD_COUNT;
    let runs_per_thread = actual_repeats / THREAD_COUNT;

    let mut threads: Vec<JoinHandle<()>> = Vec::new();
    let running = Arc::new(Mutex::new(0.0));
    let finished = Arc::new(AtomicUsize::new(0));

    for _ in 0..THREAD_COUNT {
        let counter = running.clone();
        let finished = finished.clone();

        let handle = thread::spawn(move || {
            for _ in 0..runs_per_thread {
                let r = sarsa::<TAKE_LAST_N>();
                let avg = (r.into_iter().sum::<u64>() as f64) / (TAKE_LAST_N as f64);
                let mut val = counter.lock().unwrap();
                *val += avg;
                let new_finished = finished.fetch_add(1, Ordering::SeqCst);
                println!("{}", new_finished);
            }
        });

        threads.push(handle);
    }

    for child in threads {
        let _ = child.join();
    }

    println!("{} repeats of {} trials", actual_repeats, TRIALS);
    println!("avg: {}", *running.lock().unwrap() / (REPEATS as f64));
}
