pub struct LimitedList<T, const LIMIT: usize> {
    contents: [T; LIMIT],
    ind: usize,
}

impl<T, const LIMIT: usize> LimitedList<T, LIMIT> {
    pub fn new() -> LimitedList<T, LIMIT> {
        LimitedList {
            contents: unsafe { std::mem::zeroed() },
            ind: 0,
        }
    }

    pub fn push(&mut self, val: T) {
        self.contents[self.ind] = val;
        self.ind += 1;
        self.ind %= LIMIT;
    }

    pub fn clean(&mut self) {
        self.contents.rotate_left(self.ind);
    }
}

pub struct LimitedListIterator<T, const LIMIT: usize> {
    ll: LimitedList<T, LIMIT>,
    ind: usize,
}

impl<T: Default, const LIMIT: usize> IntoIterator for LimitedList<T, LIMIT> {
    type Item = T;

    type IntoIter = LimitedListIterator<T, LIMIT>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { ll: self, ind: 0 }
    }
}

impl<T: Default, const LIMIT: usize> Iterator for LimitedListIterator<T, LIMIT> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ind < LIMIT {
            let mut v = Self::Item::default();
            std::mem::swap(&mut v, &mut self.ll.contents[self.ind]);
            self.ind += 1;
            Some(v)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_reorder() {
        let mut ll: LimitedList<usize, 4> = LimitedList::new();
        for i in 0..7 {
            ll.push(i);
        }
        assert_eq!(ll.contents, [4, 5, 6, 3]);
        ll.clean();
        assert_eq!(ll.contents, [3, 4, 5, 6]);
    }
}
