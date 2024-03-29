pub struct LimitedList<T, const LIMIT: usize> {
    contents: [T; LIMIT],
    ind: usize,
    written: usize,
}

impl<T, const LIMIT: usize> LimitedList<T, LIMIT> {
    pub fn new() -> LimitedList<T, LIMIT> {
        LimitedList {
            contents: unsafe { std::mem::zeroed() },
            ind: 0,
            written: 0,
        }
    }

    pub fn push(&mut self, val: T) {
        self.contents[self.ind] = val;
        self.ind += 1;
        self.ind %= LIMIT;
        self.written += 1;
    }

    pub fn clean(&mut self) {
        if self.written > LIMIT {
            self.contents.rotate_left(self.ind);
            self.ind = 0;
        }
    }
}

pub struct LimitedListIterator<T, const LIMIT: usize> {
    ll: LimitedList<T, LIMIT>,
    ind: usize,
}

impl<T: Default, const LIMIT: usize> IntoIterator for LimitedList<T, LIMIT> {
    type Item = T;

    type IntoIter = LimitedListIterator<T, LIMIT>;

    fn into_iter(mut self) -> Self::IntoIter {
        self.clean();
        Self::IntoIter { ll: self, ind: 0 }
    }
}

impl<T: Default, const LIMIT: usize> Iterator for LimitedListIterator<T, LIMIT> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ind < LIMIT && self.ind < self.ll.written {
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

    #[test]
    fn partial_iterator_read_out_for_non_filled_list() {
        let mut ll: LimitedList<usize, 4> = LimitedList::new();
        ll.push(1);
        ll.push(2);
        let items = ll.into_iter().collect::<Vec<_>>();
        assert_eq!(items, vec![1, 2])
    }
}
